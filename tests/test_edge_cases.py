import logging
import numpy as np
import pymupdf
import pytest
from PIL import Image

from src.utils.exception import (
    AIServiceException,
    DatabaseException,
    ImageProcessingException,
    ModelException,
)


def make_noise_image(path, size=(120, 120), mode="RGB"):
    """Random image, so it is not rejected as blank."""
    rng = np.random.default_rng(0)

    if mode == "L":
        data = rng.integers(0, 256, size, dtype=np.uint8)
    else:
        data = rng.integers(0, 256, (size[0], size[1], 3), dtype=np.uint8)

    Image.fromarray(data, mode).save(path)
    return path


# ---------------------------------------------------------------- preprocessing

def test_rgb_order_is_kept(tmp_path):
    from src.pipeline.preprocessing import preprocess_image

    # strong red channel, almost no blue. If BGR leaks through, blue ends up higher.
    rng = np.random.default_rng(1)
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[..., 0] = rng.integers(200, 256, (100, 100))
    data[..., 2] = rng.integers(0, 40, (100, 100))
    path = tmp_path / "red.png"
    Image.fromarray(data).save(path)

    image = preprocess_image(path)

    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8
    assert image[..., 0].mean() > image[..., 2].mean()


def test_non_ascii_folder_name(tmp_path):
    from src.pipeline.preprocessing import preprocess_image

    folder = tmp_path / "रोगी_scan_é"
    folder.mkdir()
    path = make_noise_image(folder / "a.png")

    assert preprocess_image(str(path)).shape == (224, 224, 3)   # plain str also works


def test_grayscale_pdf(tmp_path):
    from src.pipeline.preprocessing import preprocess_image

    image_path = make_noise_image(tmp_path / "gray.png", mode="L")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, filename=str(image_path))
    pdf_path = tmp_path / "scan.pdf"
    document.save(pdf_path)
    document.close()

    assert preprocess_image(pdf_path).shape == (224, 224, 3)


def test_corrupted_image_keeps_message(tmp_path):
    from src.pipeline.preprocessing import preprocess_image

    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")

    with pytest.raises(ImageProcessingException, match="corrupted"):
        preprocess_image(path)


def test_empty_file(tmp_path):
    from src.pipeline.preprocessing import preprocess_image

    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    with pytest.raises(ImageProcessingException, match="empty"):
        preprocess_image(path)


def test_blank_and_tiny_images(tmp_path):
    from src.pipeline.preprocessing import preprocess_image

    Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(tmp_path / "blank.png")
    make_noise_image(tmp_path / "tiny.png", size=(20, 20))

    with pytest.raises(ImageProcessingException, match="blank"):
        preprocess_image(tmp_path / "blank.png")

    with pytest.raises(ImageProcessingException, match="too small"):
        preprocess_image(tmp_path / "tiny.png")


def test_unsupported_extension(tmp_path):
    from src.pipeline.preprocessing import preprocess_image

    path = tmp_path / "scan.gif"
    path.write_bytes(b"GIF89a")

    with pytest.raises(ImageProcessingException, match="Unsupported"):
        preprocess_image(path)


# ---------------------------------------------------------------- dataset loader

def build_dataset(root, classes, train_count=12, test_count=3):
    for split, count in (("Training", train_count), ("Testing", test_count)):
        for class_name in classes:
            folder = root / split / class_name
            folder.mkdir(parents=True)

            for i in range(count):
                make_noise_image(folder / f"{i}.png", size=(100, 100))


def test_load_dataset_returns_seven_values(tmp_path, monkeypatch):
    from src.training import dataset_loader

    classes = ["Normal", "Pneumonia"]
    build_dataset(tmp_path / "Pneumonia", classes)

    # one broken file must be skipped, not stop the whole load
    (tmp_path / "Pneumonia" / "Training" / "Normal" / "bad.jpg").write_bytes(b"xx")
    monkeypatch.setattr(dataset_loader, "DATASETS_DIR", tmp_path)

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = dataset_loader.load_dataset("Pneumonia")

    assert class_names == classes
    assert X_train.dtype == np.uint8
    assert X_train.shape[1:] == (224, 224, 3)
    assert len(X_train) + len(X_val) == 24          # 2 classes x 12, bad.jpg skipped
    assert len(X_test) == 6
    assert set(np.unique(y_val)) == {0, 1}          # stratified, both classes in validation


def test_missing_class_folder_message(tmp_path, monkeypatch):
    from src.training import dataset_loader

    build_dataset(tmp_path / "Pneumonia", ["Normal"])      # Pneumonia class missing
    monkeypatch.setattr(dataset_loader, "DATASETS_DIR", tmp_path)

    with pytest.raises(ImageProcessingException, match="Class folder not found"):
        dataset_loader.load_dataset("Pneumonia")


# ---------------------------------------------------------------- risk analysis

def test_low_confidence_needs_review():
    from src.inference.riskanalyser import analyze_risk

    low = analyze_risk({"disease": "Pneumonia", "predicted_class": "Normal", "confidence": 55.0})
    high = analyze_risk({"disease": "Pneumonia", "predicted_class": "Normal", "confidence": 95.0})

    assert low["needs_review"] is True and low["risk_category"] == "Low"
    assert high["needs_review"] is False and high["confidence_level"] == "High"


def test_unknown_class_keeps_message():
    from src.inference.riskanalyser import analyze_risk

    with pytest.raises(ModelException, match="Unknown prediction class"):
        analyze_risk({"disease": "Pneumonia", "predicted_class": "Flu", "confidence": 80.0})


# ---------------------------------------------------------------- database

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from src.database import db_handler

    monkeypatch.setattr(db_handler, "DATABASE_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_handler, "database_ready", False)


def test_fresh_database_has_tables(temp_db):
    from src.database.history_manager import get_prediction_history
    from src.database.report_storage import get_reports

    # no table was created by hand, this used to fail with "no such table"
    assert get_prediction_history() == []
    assert get_reports() == []


def test_history_save_order_and_delete(temp_db):
    from src.database.history_manager import save_prediction, get_prediction_history, delete_prediction

    first = save_prediction("Pneumonia", "Normal", 90.0, "a.png")
    second = save_prediction("Pneumonia", "Pneumonia", 80.0, "b.png")

    history = get_prediction_history()
    assert [row[0] for row in history] == [second, first]       # newest first, even in the same second
    assert len(get_prediction_history(limit=1)) == 1

    assert delete_prediction(first) is True
    assert delete_prediction(first) is False                    # already gone


def test_empty_report_is_rejected(temp_db):
    from src.database.report_storage import save_report

    with pytest.raises(DatabaseException, match="empty"):
        save_report("Pneumonia", "Normal", "   ", "a.png")

    with pytest.raises(DatabaseException, match="empty"):
        save_report("Pneumonia", "Normal", None, "a.png")


# ---------------------------------------------------------------- uploads

def test_upload_name_is_never_trusted(tmp_path, monkeypatch):
    from src.pipeline import upload_handler

    monkeypatch.setattr(upload_handler, "UPLOADS_DIR", tmp_path / "uploads")

    first = upload_handler.save_upload("../../evil.png", b"data")
    second = upload_handler.save_upload("../../evil.png", b"data")

    assert first.parent == tmp_path / "uploads"
    assert first != second
    assert "evil" not in first.name


def test_upload_limits(tmp_path, monkeypatch):
    from src.pipeline import upload_handler

    monkeypatch.setattr(upload_handler, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(upload_handler, "MAX_UPLOAD_SIZE_MB", 1)

    with pytest.raises(ImageProcessingException, match="too large"):
        upload_handler.save_upload("scan.png", b"0" * (1024 * 1024 + 1))

    with pytest.raises(ImageProcessingException, match="empty"):
        upload_handler.save_upload("scan.png", b"")

    with pytest.raises(ImageProcessingException, match="Unsupported"):
        upload_handler.save_upload("notes.txt", b"hello")


# ---------------------------------------------------------------- AI services

class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeClient:
    def __init__(self, text):
        self.calls = 0
        self.models = self
        self._text = text

    def generate_content(self, model, contents):
        self.calls += 1
        return FakeResponse(self._text)


def test_ai_client_imports_without_key_and_reports_it(monkeypatch):
    from src.ai_services import ai_client

    monkeypatch.setattr(ai_client, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(ai_client, "client", None)

    with pytest.raises(AIServiceException, match="GOOGLE_API_KEY"):
        ai_client.generate_ai_report("hello")


def test_empty_ai_response_is_an_error(monkeypatch):
    from src.ai_services import ai_client

    fake = FakeClient(None)                      # what a blocked answer looks like
    monkeypatch.setattr(ai_client, "get_client", lambda: fake)
    monkeypatch.setattr(ai_client.time, "sleep", lambda seconds: None)

    with pytest.raises(AIServiceException):
        ai_client.generate_ai_report("hello")

    assert fake.calls == 3                       # retried before giving up


def test_report_always_has_disclaimer(monkeypatch):
    from src.ai_services import report_analyser

    monkeypatch.setattr(report_analyser, "generate_ai_report", lambda prompt: "Short report.")

    report = report_analyser.generate_medical_report("Pneumonia", "Normal", 90.0)

    assert report.startswith("Short report.")
    assert "not a medical diagnosis" in report


def test_prompt_mentions_low_confidence():
    from src.ai_services.prompt_builder import build_prompt

    assert "confidence is low" in build_prompt("Pneumonia", "Normal", 50.0)
    assert "confidence is low" not in build_prompt("Pneumonia", "Normal", 95.0)


# ---------------------------------------------------------------- logger

def test_logger_gets_handlers_even_if_root_has_one():
    from src.utils.logger import get_logger

    logging.getLogger().addHandler(logging.NullHandler())      # like streamlit / tensorflow do

    new_logger = get_logger("handler_check")

    assert len(new_logger.handlers) == 2
    assert new_logger.propagate is False

from importlib.metadata import packages_distributions


def test_opencv_has_only_headless_provider() -> None:
    # Competing providers overwrite cv2 and can introduce GUI library dependencies.
    providers = packages_distributions().get("cv2", [])
    assert providers == ["opencv-python-headless"]

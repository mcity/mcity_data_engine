import sys
import fiftyone as fo
from config.config import V51_ADDRESS, V51_PORT


def main():
    dataset = None
    if len(sys.argv) > 1:
        try:
            dataset = fo.load_dataset(sys.argv[1])
        except Exception:
            pass
    session = fo.launch_app(dataset=dataset, address=V51_ADDRESS, port=V51_PORT)
    session.wait(-1)


if __name__ == "__main__":
    main()
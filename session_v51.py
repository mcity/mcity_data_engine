import fiftyone as fo

from config.config import V51_ADDRESS, V51_PORT


def main():
    session = fo.launch_app(address=V51_ADDRESS, port=V51_PORT)
    session.wait(-1)  # (-1) forever


if __name__ == "__main__":
    main()

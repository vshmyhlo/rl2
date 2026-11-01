FROM python:3.12-slim-bookworm
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
RUN python -c "import sympy, z3; assert sympy.__version__ == '1.14.0'; assert z3.get_version_string() == '4.15.4'"

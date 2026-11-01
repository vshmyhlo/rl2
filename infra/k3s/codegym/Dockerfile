FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir uv==0.12.21 datasets==5.1.0
COPY src/rl2/codegym_setup.py /opt/codegym_setup.py
RUN python /opt/codegym_setup.py --root /opt/codegym --worker-host 0.0.0.0 \
    && rm -rf /root/.cache
COPY src/rl2/codegym_launcher.py /opt/codegym_launcher.py
ENV PATH="/opt/codegym/.venv/bin:${PATH}" \
    PYTHONPATH=/opt/codegym \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
WORKDIR /opt/codegym
USER 1000:1000
ENTRYPOINT ["python", "/opt/codegym_launcher.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "40"]

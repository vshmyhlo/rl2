FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir uv==0.12.21 datasets==5.1.0
COPY src/rl2/lm_envs/codegym /opt/rl2/lm_envs/codegym
RUN PYTHONPATH=/opt python -m rl2.lm_envs.codegym.setup --root /opt/codegym --worker-host 0.0.0.0 \
    && rm -rf /root/.cache
ENV PATH="/opt/codegym/.venv/bin:${PATH}" \
    PYTHONPATH=/opt:/opt/codegym \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
WORKDIR /opt/codegym
USER 1000:1000
ENTRYPOINT ["python", "-m", "rl2.lm_envs.codegym.launcher"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "40"]

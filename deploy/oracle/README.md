# Oracle Always Free backend

This deploys the API and its local index on an Ubuntu ARM VM. The frontend remains a static Vercel project.

## Provision the VM

Create an Always Free Ampere A1 VM in your home region (up to 2 OCPUs and 12 GB RAM). Give the boot volume at least 80 GB. Configure the VCN/security list to allow inbound TCP 22, 80, and 443; keep port 8000 private to localhost.

Connect over SSH, install Python, Nginx, and Git, then place the project source under `/opt/indgov`. The `.index/` directory is generated on the VM and should not be copied from a laptop.

```sh
sudo apt update
sudo apt install -y python3-venv python3-pip nginx
sudo useradd --system --home /opt/indgov --shell /usr/sbin/nologin indgov
sudo mkdir -p /opt/indgov /var/cache/indgov/huggingface
sudo chown -R indgov:indgov /opt/indgov /var/cache/indgov
```

Copy the project files to `/opt/indgov`, then create `/etc/indgov.env` with the exact Vercel production origin (you can add it after the first frontend deployment):

```sh
sudo install -m 600 /dev/null /etc/indgov.env
sudoedit /etc/indgov.env
```

Contents:

```ini
INDGOV_CORS_ORIGINS=https://your-vercel-domain.vercel.app
```

Log in to Hugging Face as the `indgov` service user so its token stays in that user's cache, outside the project tree. First accept access to `google/embeddinggemma-300m` on Hugging Face in your browser.

```sh
sudo -u indgov env HF_HOME=/var/cache/indgov/huggingface /opt/indgov/.venv/bin/hf auth login
```

Paste your token at the interactive prompt.

Build the Python environment and dataset index as the service user. This downloads the public embedded Parquet and creates about 4.5 GB of index files on the VM. It can take a while.

```sh
sudo -u indgov python3 -m venv /opt/indgov/.venv
sudo -u indgov /opt/indgov/.venv/bin/pip install -r /opt/indgov/requirements.txt
sudo -u indgov env HF_HOME=/var/cache/indgov/huggingface \
  /opt/indgov/.venv/bin/python /opt/indgov/build_index.py --index-dir /opt/indgov/.index
sudo -u indgov env HF_HOME=/var/cache/indgov/huggingface \
  /opt/indgov/.venv/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("google/embeddinggemma-300m", device="cpu")'
```

The model is cached under `/var/cache/indgov/huggingface` so the service can start without downloading it again.

## Start the API

```sh
sudo cp /opt/indgov/deploy/oracle/indgov.service /etc/systemd/system/indgov.service
sudo systemctl daemon-reload
sudo systemctl enable --now indgov
sudo cp /opt/indgov/deploy/oracle/nginx.conf /etc/nginx/sites-available/indgov
sudo ln -sf /etc/nginx/sites-available/indgov /etc/nginx/sites-enabled/indgov
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Nginx initially serves HTTP on port 80. Point a hostname at the VM public IP, then install a Let's Encrypt certificate with Certbot and enable HTTPS. Set `INDGOV_CORS_ORIGINS` to the exact Vercel production origin and restart `indgov` after the frontend URL is known. The Vercel frontend must call the HTTPS API URL.

Check readiness at `http://VM-HOSTNAME/api/health`. It should report both `index_ready` and `model_ready` as `true`.

## Vercel frontend

Import the project repository into Vercel with the project root set to the repo root. Add the environment variable `INDGOV_API_URL=https://VM-HOSTNAME/api`, then deploy. The build command in `vercel.json` writes the API URL into the static frontend. After Vercel gives the production URL, set that exact origin in `/etc/indgov.env` as `INDGOV_CORS_ORIGINS` on the VM and restart the API.

Vercel needs the source in a Git repository. Keep `.index/`, `.venv/`, HF credentials, and `.env` files out of that repository.

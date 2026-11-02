# Advanced Setup & Deployment

> **v2 (legacy).** This page describes the v2 application (the RAG chatbot) on the `main` branch. It is not the 3.0 documentation. v2 keeps running on `main` until the cutover; see [Moving from v2 to 3.0](../migration-from-v2.md).

Topics related to advanced setup and deployment of Archi.

## Configuring Podman

To ensure your Podman containers stay running for extended periods, you need to enable lingering. To do this, run:

```bash
loginctl enable-linger
```

To check or confirm the lingering status, run:

```bash
loginctl user-status | grep -m1 Linger
```

See the Red Hat [documentation](https://access.redhat.com/solutions/7054698) for additional context.

## Running LLMs locally on your GPUs

There are a few additional system requirements for this to work:

1. Make sure you have NVIDIA drivers installed.
2. (Optional) For the containers where Archi will run to access the GPUs, install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Configure the container runtime to access the GPUs.

<details>
<summary>For Podman</summary>

Run the following command:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Then list the devices:

```bash
nvidia-ctk cdi list
```

You should see output similar to:

```text
INFO[0000] Found 9 CDI devices
...
nvidia.com/gpu=0
nvidia.com/gpu=1
...
nvidia.com/gpu=all
...
```

These listed "CDI devices" will be referenced to run Archi on the GPUs, so make sure they are present. To learn more, consult the [Podman GPU documentation](https://podman-desktop.io/docs/podman/gpu).

</details>

<details>
<summary>For Docker</summary>

Run the following command:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

The remaining steps mirror the Podman flow. NOTE: this has not yet been fully tested with Docker. Refer to the [NVIDIA toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration) for details.

</details>

Once these requirements are met, the `archi create [...] --gpu-ids <gpus>` option will deploy Archi across your GPUs. For Helm deployments, `archi install [...] --gpu-ids <gpus>` writes the GPU settings into the generated chart. Explicit IDs request the matching `nvidia.com/gpu` count; `all` sets `NVIDIA_VISIBLE_DEVICES=all`, and you can set `gpu.count` in `values.yaml` when your Kubernetes cluster requires an explicit limit.

## Helpful Notes for Production Deployments

You may wish to use the CLI in order to stage production deployments. This section covers some useful notes to keep in mind.

### Running multiple deployments on the same machine

The CLI allows multiple deployments to run on the same daemon in the case of Docker (Podman has no daemon). The container networks between all the deployments are separate, so there is very little risk of them accidentally communicating with one another.

However, you need to be careful with the external ports. Suppose you're running two deployments and both of them are running the chat on external port 8000. There is no way to view both deployments at the same time from the same port, so instead you should forward the deployments to other external ports. Generally, this can be done in the configuration:

```yaml
services:
  chat_app:
    external_port: 7862  # default is 7861
  uploader_app:
    external_port: 5004  # default is 5003
  grafana:
    external_port: 3001  # default is 3000
  postgres:
    port: 5432  # default is 5432
```

### Persisting data between deployments

Volumes persist between deployments, so if you deploy an instance and upload additional documents, you do not need to redo this every time you deploy. If you are editing any data, explicitly remove this information from the volume, or remove the volume itself with:

```bash
docker/podman volume rm <volume-name>
```

To see what volumes are currently present, run:

```bash
docker/podman volume ls
```

### HTTPS Configuration for Production

For production deployments, especially when using BYOK (Bring Your Own Key), HTTPS is strongly recommended to protect API keys in transit.

#### Using a Reverse Proxy

The recommended approach is to terminate TLS at a reverse proxy (nginx, Caddy, Traefik):

**Example nginx configuration:**

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:7861;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE streaming support
        proxy_buffering off;
        proxy_cache off;
    }
}
```

**Example Caddy configuration (automatic HTTPS):**

```caddyfile
your-domain.com {
    reverse_proxy localhost:7861
}
```

#### Session Cookie Security

When running behind HTTPS, enable secure cookies by setting the environment variable:

```bash
FLASK_SESSION_COOKIE_SECURE=true
```

Or configure in your deployment's environment file:

```env
FLASK_SESSION_COOKIE_SECURE=true
```

This ensures session cookies (which may contain API keys) are only sent over encrypted connections.

#### Security Checklist

- [ ] TLS termination at reverse proxy or load balancer
- [ ] `FLASK_SESSION_COOKIE_SECURE=true` in production
- [ ] Strong `FLASK_UPLOADER_APP_SECRET_KEY` configured (not auto-generated)
- [ ] Firewall rules limiting direct access to internal ports
- [ ] Regular certificate renewal (use Let's Encrypt/certbot)

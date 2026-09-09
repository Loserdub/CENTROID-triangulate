# Security Policy

## Responsible Disclosure

If you discover a security vulnerability in CENTROID, please report it responsibly rather than opening a public issue on GitHub.

To report a vulnerability:
- Use the **Private Vulnerability Reporting** feature in the GitHub repository (navigate to the **Security** tab and select **Report a vulnerability**).
- Provide a detailed description of the vulnerability, reproduction steps, and potential impact.

---

## Security Considerations for Local Analysis

1. **Private Corpora & Data Confidentiality**:
   - CENTROID executes entirely on your local machine. It does not transmit document text, extracted vocabulary, or analytical outputs to external servers, cloud LLMs, or telemetry endpoints.
   - When analyzing sensitive, proprietary, or classified documents, ensure your output directory (`outputs/`) and intermediate files are protected in accordance with your organization's data retention policies.

2. **Untrusted Web Ingestion**:
   - Stage 01 (`pipeline/01_fetch.py`) downloads content from URLs specified in `inputs/urls.txt`.
   - Streaming requests enforce a 10MB response ceiling (`MAX_RESPONSE_BYTES`) and validate `Content-Type` headers (`text/html`, `text/plain`) to mitigate denial-of-service and memory exhaustion risks.
   - Avoid configuring URLs to untrusted internal network endpoints (e.g. metadata services like `http://169.254.169.254`).

3. **HTML Dashboard & Injection Prevention**:
   - All text extracted from documents and injected into `outputs/dashboard.html` is escaped via `html.escape()` and JSON serialized with script-closing tag sanitization (`safe_json_dumps`).
   - If embedding external untrusted data into custom templates, ensure escaping mechanisms remain intact.

4. **Public Submissions**:
   - Do not commit confidential corporate dossiers, sensitive personal data, API keys, or private documents to public pull requests, issues, or fork repositories.

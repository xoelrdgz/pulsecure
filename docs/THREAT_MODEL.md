# Threat Model

Local PoC for encrypted cardiovascular screening. The browser, Rust binary, and SQLite database run on the same machine.

## Assets

- User-entered data
- Local encryption key
- Stored screenings
- Signed model
- Audit history

## Controls

- Signed model verification before load
- FHE computation through tfhe-rs
- Clinical result encryption in SQLite
- Patient reference pseudonymization
- HTTP security headers
- Container with `no-new-privileges`, read-only filesystem, and Docker secret for the local password

## Out Of Scope

- Real clinical validation
- Regulatory compliance
- Multi-user operation
- Internet-facing remote deployment
- Advanced identity management

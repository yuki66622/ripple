# Public release scope — 2026-09-27

User explicitly authorized a new public GitHub repository and publication of this project.

- Root owns source snapshot, requirements, model setup, tests, privacy inspection, GitHub creation and push.
- release_runtime_audit owns only README.md and THIRD_PARTY_NOTICES.md in this directory.
- release_research_audit remains read-only.
- Public copy is separate from the working project. Exclude credentials, raw market archives, weights, runtime outputs, machine-specific notes, caches, private screenshots and existing Git history.
- Include current frontend/backend, deterministic metrics, evaluation and training source, selected frozen research results, cropped demo video, reproducible setup and truthful experiment limitations.
- Do not run training, new forecasting benchmarks, or open untouched raw holdouts during release.

Latest user instruction: include what was tested, the logic, actual results, and failures. release_research_audit now owns only docs/EXPERIMENTS.md; root owns copying/sanitizing supporting frozen protocols/reports. Do not turn exploratory results into claims of sealed-test success.

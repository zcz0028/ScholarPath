# ScholarPath Day8-3 Frontend Evidence & Recommendation Integration

Replaced:
- apps/web/src/types/api.ts
- apps/web/src/components/PaperDetails.tsx

Added:
- apps/web/src/components/PaperDetails.day8.css
- docs/week2_day8_3_frontend_evidence.md
- README_DAY8_3.md

Run:
cd apps\web
npm run build

Then:
npm run dev

Acceptance:
- Why Recommended shows backend reason_text and matched/unmatched constraints.
- Match Evidence renders structured cards instead of JSON.
- openalex_live is shown as OpenAlex Live Retrieval / OpenAlex 实时检索.
- citation_path=null never creates a synthetic citation path.
- UI strings in PaperDetails.tsx are valid UTF-8.

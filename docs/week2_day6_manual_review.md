# Week 2 Day 6 Manual Review

## Git

- Branch: `week2/competition-api`
- Commit:
- Test result:

## Environment

- Python:
- FastAPI:
- Uvicorn:
- Pydantic:
- HTTPX:

## Endpoints

- [ ] `GET /health`
- [ ] `GET /api/system`
- [ ] `GET /api/queries`
- [ ] `POST /api/search`
- [ ] `GET /api/queries/{qid}/diagnosis`
- [ ] `GET /api/experiments`
- [ ] Swagger `/docs`

## Benchmark Mode

- Query count: 50
- Search qid:
- Top K:
- Returned results:
- Gold fields exposed: false
- Benchmark API calls: 0

## Diagnosis

- [ ] Constraints available
- [ ] Academic anchors available
- [ ] Query plan available
- [ ] Rescue trace available
- [ ] Citation trace available
- [ ] Cost summary available

## Live Mode

- Query:
- Planned query count:
- OpenAlex API calls:
- Cache hits:
- Returned results:
- Citation expansion default off: true

## Error Handling

- [ ] Unknown qid returns 404
- [ ] Invalid top_k returns structured 422
- [ ] Missing artifact returns structured error
- [ ] No traceback exposed

## Competition Demo Notes

- Benchmark mode is the primary judged/demo path because it exactly matches frozen offline evaluation.
- Live mode proves the system supports unseen natural-language queries.
- Day-5 citation paths are shown as explainable discovery evidence, not as unvalidated final recommendations.

## Conclusion

Day 6 accepted / rejected.

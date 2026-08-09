# Week 2 Day 7 Manual Review

## Git

- Branch: week2/competition-frontend
- Commit:

## Environment

- Node:
- npm:
- Backend URL: http://127.0.0.1:8000
- Frontend URL: http://127.0.0.1:5173

## Build

- [ ] `npm install` successful
- [ ] `npm run build` successful
- [ ] FastAPI CORS works from port 5173

## Benchmark Mode

- [ ] `/api/queries` loads real queries
- [ ] User can select a real qid without seeing engineering qid in normal flow
- [ ] `POST /api/search` returns 200
- [ ] Paper list uses real API results
- [ ] No fixed 0.92 / 82% / fake cost values
- [ ] Search Reasoning uses real constraints/plans/anchors/pipeline

## Paper Details

- [ ] Initial state does not auto-select Rank 1
- [ ] Clicking a paper opens details
- [ ] Why Recommended uses real `reason_text/reason_tags`
- [ ] Match Evidence uses real evidence / score only
- [ ] Citation Path disabled when `citation_path = null`
- [ ] Retrieval Sources come from API
- [ ] View Paper opens real URL

## Live Mode

- [ ] Arbitrary query can be entered
- [ ] qid is not required
- [ ] Loading/error state works
- [ ] Backend warnings are shown without pretending Live is evaluated Benchmark output

## Language

- [ ] 中文 / EN switches UI copy
- [ ] Paper titles/authors/query/venue remain source-language metadata

## Conclusion

Day 7 accepted / rejected.

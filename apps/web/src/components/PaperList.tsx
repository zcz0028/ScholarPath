import { ExternalLink } from "lucide-react";
import { useMemo } from "react";
import type { Language, PaperResult } from "../types/api";
import { useCopy } from "../i18n/translations";

interface PaperListProps {
  language: Language;
  papers: PaperResult[];
  selectedPaper: PaperResult | null;
  onSelectPaper: (paper: PaperResult) => void;
}

export function PaperList({
  language,
  papers,
  selectedPaper,
  onSelectPaper,
}: PaperListProps) {
  const copy = useCopy(language);

  /*
   * Day14 UI display ordering.
   *
   * IMPORTANT:
   * - Does NOT mutate response.results.
   * - Does NOT modify backend paper.rank.
   * - Does NOT modify Day8 E3 artifacts or evaluation.
   * - Only controls presentation order in this component.
   *
   * Primary key:
   *   score descending
   *
   * Tie break:
   *   frozen backend rank ascending
   *
   * Missing scores:
   *   placed after scored papers.
   */
  const displayedPapers = useMemo(() => {
    return papers
      .map((paper, originalIndex) => ({
        paper,
        originalIndex,
      }))
      .sort((left, right) => {
        const leftScore =
          typeof left.paper.score === "number"
            ? left.paper.score
            : Number.NEGATIVE_INFINITY;

        const rightScore =
          typeof right.paper.score === "number"
            ? right.paper.score
            : Number.NEGATIVE_INFINITY;

        if (Math.abs(leftScore - rightScore) > 1e-12) {
          return rightScore - leftScore;
        }

        if (left.paper.rank !== right.paper.rank) {
          return left.paper.rank - right.paper.rank;
        }

        return left.originalIndex - right.originalIndex;
      })
      .map((item) => item.paper);
  }, [papers]);

  return (
    <section className="paper-list-card">
      <div className="list-toolbar">
        <span>
          <strong>{papers.length}</strong> {copy.results}
        </span>

        <span>
          {copy.sort}:{" "}
          <strong>
            {language === "zh"
              ? "得分从高到低"
              : "Score: High to Low"}
          </strong>
        </span>
      </div>

      <div className="paper-list">
        {displayedPapers.length === 0 && (
          <div className="paper-empty">{copy.noResults}</div>
        )}

        {displayedPapers.map((paper, displayIndex) => {
          const selected = isSamePaper(selectedPaper, paper);

          return (
            <article
              key={paperKey(paper)}
              className={`paper-row ${selected ? "selected" : ""}`}
              onClick={() => onSelectPaper(paper)}
            >
              {/*
               * This badge is the DISPLAY rank after score sorting.
               * paper.rank itself is intentionally left untouched.
               */}
              <div
                className="rank-badge"
                title={
                  language === "zh"
                    ? `显示排名 ${displayIndex + 1}；后端原排名 ${paper.rank}`
                    : `Display rank ${displayIndex + 1}; backend rank ${paper.rank}`
                }
              >
                {displayIndex + 1}
              </div>

              <div className="paper-main">
                <h2>{paper.title}</h2>

                <p className="paper-authors">
                  {paper.authors.length
                    ? paper.authors.slice(0, 6).join(", ") +
                    (paper.authors.length > 6 ? ", et al." : "")
                    : "—"}
                </p>

                <p className="paper-meta">
                  {[paper.year, paper.venue]
                    .filter(Boolean)
                    .join(" · ")}
                </p>

                <div className="paper-tags">
                  {paper.reason_tags.slice(0, 5).map((tag) => (
                    <span key={tag}>
                      {formatTag(tag, language)}
                    </span>
                  ))}

                  {paper.openalex_id && <span>OpenAlex</span>}
                  {paper.arxiv_id && <span>arXiv</span>}
                  {paper.doi && <span>DOI</span>}
                </div>
              </div>

              <div className="paper-side">
                <div
                  className="score-box"
                  title={
                    paper.score == null
                      ? undefined
                      : `Raw score: ${paper.score}`
                  }
                >
                  <strong>
                    {paper.score == null
                      ? "—"
                      : paper.score.toFixed(3)}
                  </strong>

                  <span>{copy.score}</span>
                </div>

                {paper.url && (
                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noreferrer"
                    onClick={(event) =>
                      event.stopPropagation()
                    }
                    aria-label={copy.viewPaper}
                  >
                    <ExternalLink size={19} />
                  </a>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function paperKey(paper: PaperResult): string {
  return (
    paper.openalex_id ||
    paper.doi ||
    paper.arxiv_id ||
    `${paper.rank}-${paper.title}`
  );
}

function isSamePaper(
  left: PaperResult | null,
  right: PaperResult,
): boolean {
  if (!left) return false;

  if (
    left.openalex_id &&
    right.openalex_id &&
    left.openalex_id === right.openalex_id
  ) {
    return true;
  }

  if (
    left.doi &&
    right.doi &&
    left.doi === right.doi
  ) {
    return true;
  }

  if (
    left.arxiv_id &&
    right.arxiv_id &&
    left.arxiv_id === right.arxiv_id
  ) {
    return true;
  }

  return (
    left.rank === right.rank &&
    left.title === right.title
  );
}

function formatTag(
  tag: string,
  language: Language,
) {
  const labels: Record<string, [string, string]> = {
    anchor_rescue_source: [
      "救援召回",
      "Rescue Retrieval",
    ],
    multi_plan_supported: [
      "多计划支持",
      "Multi-plan Support",
    ],
    high_rank_rescue_hit: [
      "高位救援命中",
      "High-rank Rescue",
    ],
    strong_identifier_available: [
      "强标识符",
      "Strong ID",
    ],
    multi_source_supported: [
      "多来源支持",
      "Multi-source Support",
    ],
    high_rank_in_at_least_one_source: [
      "来源高排名",
      "High Rank in Source",
    ],
  };

  const mapped = labels[tag];

  return mapped
    ? mapped[language === "zh" ? 0 : 1]
    : tag.replace(/_/g, " ");
}
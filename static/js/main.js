document.addEventListener("DOMContentLoaded", function () {

    const recommendBtn = document.getElementById("recommendBtn");
    const resultsDiv   = document.getElementById("results");

    // =====================================================
    // CARD GENERATOR
    // =====================================================
    const createCourseCard = (course, query, model, isCompact = false) => {
        const scoreColor = course.score > 70 ? '#134074' : '#8da9c4';
        return `
            <div class="${isCompact ? 'col-12' : 'col-md-6 col-lg-4'} mb-4">
                <div class="card h-100 border-0 shadow-sm card-hover bg-white rounded-4 overflow-hidden">
                    <div class="card-body p-4">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <span class="badge rounded-pill px-3 py-1 mb-2"
                                  style="background-color: #eef4ed; color: #134074; border: 1px solid #8da9c4;">
                                ${course.department}
                            </span>
                            <span class="fw-bold" style="color: ${scoreColor}">${course.score}% Match</span>
                        </div>
                        <h5 class="card-title fw-bold text-brand-deep">${course.title}</h5>
                        <p class="card-text text-muted small mb-4">${course.description.substring(0, 120)}...</p>

                        <div class="progress mb-4" style="height: 8px; background-color: #eef4ed;">
                            <div class="progress-bar rounded-pill" role="progressbar"
                                 style="width: ${course.score}%; background-color: ${scoreColor};">
                            </div>
                        </div>

                        ${!isCompact ? `
                        <div class="d-flex gap-2">
                            <button class="btn btn-brand w-100 rounded-3 py-2 save-btn"
                                    data-title="${course.title}"
                                    data-department="${course.department}"
                                    data-score="${course.score}"
                                    data-query="${query}"
                                    data-model="${model}"
                                    data-url="${course.url || ''}">
                                    <i class="bi bi-bookmark-plus"></i> Save
                            </button>
                            <button class="btn btn-outline-secondary rounded-3 py-2 view-btn"
                                    data-url="${course.url || '#'}"
                                    style="white-space: nowrap;">
                                    <i class="bi bi-box-arrow-up-right"></i> View
                            </button>
                        </div>` : ''}
                    </div>
                </div>
            </div>`;
    };

    // =====================================================
    // SAVE BUTTON BINDER
    // Now includes url in POST body for dashboard linking
    // =====================================================
    const bindSaveButtons = () => {
        document.querySelectorAll(".save-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                await fetch("/api/save", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        query:        btn.dataset.query,
                        model:        btn.dataset.model,
                        course_title: btn.dataset.title,
                        subject:      btn.dataset.department,
                        score:        parseFloat(btn.dataset.score),
                        url:          btn.dataset.url || ""
                    })
                });
                btn.innerHTML = "✓ Saved";
                btn.classList.replace("btn-brand", "btn-success");
                btn.disabled = true;
            });
        });
    };

    // =====================================================
    // VIEW BUTTON BINDER
    // =====================================================
    const bindViewButtons = () => {
        document.querySelectorAll(".view-btn").forEach(btn => {
            btn.addEventListener("click", (event) => {
                event.stopPropagation();
                const url = btn.dataset.url;
                if (url && url !== "#") {
                    window.open(url, "_blank", "noopener,noreferrer");
                }
            });
        });
    };

    // =====================================================
    // SINGLE MODEL RENDERER
    // =====================================================
    const renderSingleModel = (results, query, model) => {
        if (!results || results.length === 0) {
            resultsDiv.innerHTML = `
                <p class='text-center text-muted py-5'>
                    No results found. Try a different query.
                </p>`;
            return;
        }

        resultsDiv.innerHTML = results
            .map(course => createCourseCard(course, query, model, false))
            .join("");

        bindSaveButtons();
        bindViewButtons();
    };

    // =====================================================
    // BOTH MODELS RENDERER
    // =====================================================
    const renderBothModels = (tfidfResults, neuralResults, query) => {
        resultsDiv.innerHTML = `
            <div class="col-12 mb-4">
                <div class="row g-4">

                    <div class="col-12 col-lg-6">
                        <div class="d-flex align-items-center mb-3 gap-2">
                            <span class="badge rounded-pill px-3 py-2"
                                  style="background-color: #134074; color: white; font-size: 0.85rem;">
                                TF-IDF Model
                            </span>
                            <small class="text-muted">Keyword-based matching</small>
                        </div>
                        <div class="row" id="tfidf-results-container">
                            ${(tfidfResults || [])
                                .map(course => {
                                    let cardHtml = createCourseCard(course, query, "tfidf", false);
                                    return cardHtml.replace('col-md-6 col-lg-4', 'col-md-6');
                                })
                                .join("")}
                        </div>
                    </div>

                    <div class="col-12 col-lg-6">
                        <div class="d-flex align-items-center mb-3 gap-2">
                            <span class="badge rounded-pill px-3 py-2"
                                  style="background-color: #0b2545; color: white; font-size: 0.85rem; font-weight: 600;">
                                Neural Model
                            </span>
                            <small class="text-muted">Semantic understanding</small>
                        </div>
                        <div class="row" id="neural-results-container">
                            ${(neuralResults || [])
                                .map(course => {
                                    let cardHtml = createCourseCard(course, query, "neural", false);
                                    return cardHtml.replace('col-md-6 col-lg-4', 'col-md-6');
                                })
                                .join("")}
                        </div>
                    </div>

                </div>
            </div>`;

        bindSaveButtons();
        bindViewButtons();
    };

    // =====================================================
    // KEYBOARD UX — RECOMMEND PAGE
    // =====================================================
    const queryInput  = document.getElementById("query");
    const modelSelect = document.getElementById("model");

    if (queryInput) {
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                if (modelSelect) {
                    modelSelect.focus();
                    modelSelect.dispatchEvent(new MouseEvent("mousedown"));
                }
            }
        });
    }

    if (modelSelect) {
        modelSelect.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                if (recommendBtn) {
                    recommendBtn.click();
                }
            }
        });
    }

    // =====================================================
    // RECOMMENDATION PAGE
    // =====================================================
    if (recommendBtn) {
        recommendBtn.addEventListener("click", async () => {
            const query = document.getElementById("query").value.trim();
            const model = document.getElementById("model").value;

            if (!query) {
                alert("Please enter a query!");
                return;
            }

            resultsDiv.innerHTML = `
                <div class="col-12 text-center py-5">
                    <div class="spinner-border text-primary" role="status"></div>
                    <p class="mt-3 text-brand-muted">AI is analyzing courses...</p>
                </div>`;

            try {
                const response = await fetch("/api/recommend", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query, model })
                });

                const data = await response.json();

                if (data.error) {
                    resultsDiv.innerHTML = `
                        <p class='text-danger text-center py-4'>${data.error}</p>`;
                    return;
                }

                if (model === "both") {
                    renderBothModels(data.tfidf_results, data.neural_results, query);
                } else if (model === "tfidf") {
                    renderSingleModel(data.tfidf_results, query, model);
                } else if (model === "neural") {
                    renderSingleModel(data.neural_results, query, model);
                }

            } catch (e) {
                resultsDiv.innerHTML = `
                    <p class='text-danger text-center py-4'>
                        Error loading recommendations. Please try again.
                    </p>`;
            }
        });
    }

    // =====================================================
    // DASHBOARD PAGE
    // =====================================================
    if (document.getElementById("historyTable")) {

        // ----------------------------
        // LOAD HISTORY
        // ----------------------------
        async function loadHistory() {
            const res   = await fetch("/api/history");
            const data  = await res.json();
            const tbody = document.querySelector("#historyTable tbody");

            if (data.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="4" class="text-center py-4 text-muted">
                            No search history yet.
                        </td>
                    </tr>`;
                return;
            }

            tbody.innerHTML = data.map(row => `
                <tr>
                    <td class="ps-4 fw-semibold text-brand-deep">${row[1]}</td>
                    <td>
                        <span class="badge bg-light text-dark border">
                            ${row[2].toUpperCase()}
                        </span>
                    </td>
                    <td class="text-muted small">
                        ${new Date(row[3]).toLocaleDateString()}
                    </td>
                    <td class="pe-4 text-end">
                        <button class="btn btn-sm btn-outline-primary rounded-pill px-3 compare-btn"
                                data-query="${row[1]}">
                            Compare AI
                        </button>
                    </td>
                </tr>`).join("");

            document.querySelectorAll(".compare-btn").forEach(btn => {
                btn.addEventListener("click", async () => {
                    const query = btn.dataset.query;

                    ["compareTFIDF", "compareNeural"].forEach(id => {
                        const el = document.getElementById(id);
                        if (el) {
                            el.innerHTML = `
                                <div class="p-3 text-center small text-muted">
                                    Consulting AI...
                                </div>`;
                        }
                    });

                    const res  = await fetch("/api/recommend", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query, model: "both" })
                    });
                    const data = await res.json();

                    const tfidfEl  = document.getElementById("compareTFIDF");
                    const neuralEl = document.getElementById("compareNeural");

                    if (tfidfEl) {
                        tfidfEl.innerHTML = (data.tfidf_results || [])
                            .slice(0, 3)
                            .map(c => createCourseCard(c, query, "tfidf", true))
                            .join("");
                    }

                    if (neuralEl) {
                        neuralEl.innerHTML = (data.neural_results || [])
                            .slice(0, 3)
                            .map(c => createCourseCard(c, query, "neural", true))
                            .join("");
                    }
                });
            });
        }

        // ----------------------------
        // LOAD SAVED RECOMMENDATIONS
        // Course title rendered as clickable link when URL exists.
        // Falls back to plain text if URL is missing or empty.
        // row indices: id(0), query(1), model(2), course_title(3),
        //              subject(4), score(5), timestamp(6), url(7)
        // ----------------------------
        async function loadSaved() {
            const res   = await fetch("/api/saved");
            const data  = await res.json();
            const tbody = document.querySelector("#savedTable tbody");

            if (data.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center py-4 text-muted">
                            No saved courses yet.
                        </td>
                    </tr>`;
                return;
            }

            tbody.innerHTML = data.map(row => `
                <tr>
                    <td class="ps-4 small text-muted">${row[1]}</td>
                    <td>
                        ${row[7] && row[7] !== ""
                            ? `<a href="${row[7]}"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  class="fw-bold text-brand-deep"
                                  style="text-decoration: none;"
                                  onmouseover="this.style.textDecoration='underline'"
                                  onmouseout="this.style.textDecoration='none'">
                                  ${row[3]}
                               </a>`
                            : `<span class="fw-bold text-brand-deep">${row[3]}</span>`
                        }
                    </td>
                    <td class="text-center">
                        <span class="badge rounded-pill bg-light text-brand-deep border">
                            ${row[4]}
                        </span>
                    </td>
                    <td class="text-center fw-bold text-brand-deep">${row[5]}%</td>
                    <td class="pe-4 text-end">
                        <button class="btn btn-sm text-danger unsave-btn"
                                data-query="${row[1]}"
                                data-title="${row[3]}">
                            Remove
                        </button>
                    </td>
                </tr>`).join("");

            document.querySelectorAll(".unsave-btn").forEach(btn => {
                btn.addEventListener("click", async () => {
                    if (!confirm(`Remove "${btn.dataset.title}"?`)) return;
                    await fetch("/api/delete_saved", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            query:        btn.dataset.query,
                            course_title: btn.dataset.title
                        })
                    });
                    loadSaved();
                });
            });
        }

        // ----------------------------
        // CLEAR HISTORY BUTTON
        // ----------------------------
        const clearHistoryBtn = document.getElementById("clearHistoryBtn");
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener("click", async () => {
                if (!confirm("Clear all search history?")) return;
                await fetch("/api/clear_history", { method: "POST" });
                loadHistory();
            });
        }

        loadHistory();
        loadSaved();

        // Auto-refresh dashboard every 5 seconds
        setInterval(() => {
            loadHistory();
            loadSaved();
        }, 5000);
    }

});
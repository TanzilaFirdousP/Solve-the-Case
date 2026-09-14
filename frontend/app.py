import os
import html

import httpx
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="The Casebook",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# BACKEND
# =========================================================

DEFAULT_BACKEND = os.getenv(
    "DETECTIVE_API_URL",
    "http://127.0.0.1:8000",
)

try:
    DEFAULT_BACKEND = st.secrets.get(
        "BACKEND_URL",
        DEFAULT_BACKEND,
    )
except Exception:
    pass

BACKEND_URL = DEFAULT_BACKEND


# =========================================================
# THEME
# =========================================================

st.html(
    """
    <style>

    :root {
        --black: #100d0a;
        --dark: #17120e;
        --panel: #211a14;
        --panel2: #2b2118;
        --brown: #654930;
        --brown-light: #896548;
        --gold: #b89762;
        --cream: #eadcc3;
        --paper: #d7c4a3;
        --muted: #a99475;
        --border: #594532;
        --green: #9caf83;
        --orange: #d09565;
    }

    html,
    body,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at top,
                #251d15 0%,
                #17120e 48%,
                #100d0a 100%
            );
        color: var(--paper);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0d0a08;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--paper);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        font-family: Georgia, "Times New Roman", serif !important;
        color: var(--cream) !important;
    }

    p, label {
        color: var(--paper);
    }

    hr {
        border-color: #49382a !important;
    }

    /* ---------- HEADER ---------- */

    .detective-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 3.15rem;
        font-weight: 700;
        color: var(--cream);
        letter-spacing: 0.02em;
        margin-bottom: 0;
    }

    .detective-tagline {
        color: var(--muted);
        font-family: Georgia, "Times New Roman", serif;
        font-style: italic;
        font-size: 1.08rem;
        margin-top: -0.15rem;
        margin-bottom: 1.8rem;
    }

    /* ---------- CASE FILE ---------- */

    .case-file {
        background: linear-gradient(
            135deg,
            #2a2118,
            #1d1712
        );

        border: 1px solid var(--border);
        border-left: 5px solid var(--gold);

        padding: 1.5rem 1.65rem;
        border-radius: 7px;

        box-shadow:
            0 14px 35px rgba(0,0,0,0.30);

        margin: 1rem 0 1.5rem 0;
    }

    .case-number {
        color: var(--gold);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .case-description {
        color: #c9b89a;
        line-height: 1.65;
        margin-top: 0.3rem;
    }

    .mission-label {
        color: var(--gold);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    .mission-text {
        color: var(--cream);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.13rem;
        line-height: 1.6;
        margin-top: 0.4rem;
    }

    /* ---------- CARDS ---------- */

    .suspect-card {
        background: #211a14;

        border:
            1px solid var(--border);

        border-top:
            3px solid #795a3e;

        padding:
            1rem;

        border-radius:
            6px;

        min-height:
            90px;

        box-shadow:
            0 6px 17px rgba(0,0,0,0.18);

        margin-bottom:
            0.7rem;
    }

    .suspect-name {
        color: var(--cream);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.08rem;
        font-weight: 700;
    }

    .suspect-type {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }

    .clue-card {
        background: #211a14;

        border:
            1px solid #49382a;

        border-left:
            4px solid var(--brown-light);

        padding:
            1rem 1.1rem;

        border-radius:
            5px;

        margin:
            0.8rem 0;

        box-shadow:
            0 5px 17px rgba(0,0,0,0.16);
    }

    .clue-title {
        color: var(--cream);
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 700;
    }

    .clue-text {
        color: #cbb99c;
        margin-top: 0.65rem;
        line-height: 1.5;
    }

    .verified-text {
        color: var(--green);
        font-weight: 700;
        font-size: 0.85rem;
    }

    .unverified-text {
        color: var(--orange);
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        background:
            linear-gradient(
                180deg,
                #77573c,
                #58402d
            ) !important;

        color:
            #f2e4cc !important;

        border:
            1px solid #94704f !important;

        border-radius:
            5px !important;

        font-weight:
            600 !important;

        transition:
            0.15s ease-in-out;
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                180deg,
                #8a6648,
                #684a33
            ) !important;

        border-color:
            var(--gold) !important;

        transform:
            translateY(-1px);
    }

    /* ---------- INPUTS ---------- */

    input,
    textarea,
    [data-baseweb="select"] > div {
        background-color:
            #1a1510 !important;

        color:
            var(--cream) !important;

        border-color:
            var(--border) !important;
    }

    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        color:
            #aa9577 !important;

        font-family:
            Georgia, "Times New Roman", serif;

        font-size:
            1rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color:
            var(--cream) !important;

        border-bottom-color:
            var(--gold) !important;
    }

    /* ---------- EXPANDERS ---------- */

    [data-testid="stExpander"] {
        background:
            #1b1611;

        border:
            1px solid #49382a;

        border-radius:
            5px;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background:
            #211a14;

        border:
            1px solid #49382a;

        padding:
            0.85rem;

        border-radius:
            6px;
    }

    [data-testid="stMetricValue"] {
        color:
            #dfcba8;
    }

    /* ---------- ALERTS ---------- */

    [data-testid="stAlert"] {
        background:
            #2a2117 !important;

        border:
            1px solid #6e5438 !important;

        color:
            #e4d3b6 !important;
    }

    [data-testid="stAlert"] * {
        color:
            #e4d3b6 !important;
    }

    </style>
    """
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "case_id": None,
    "board": {},
    "search_results": [],
    "interrogation": None,
    "prediction": None,
    "investigation": None,
    "fact_check": None,
    "verdict": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


def reset_case():

    st.session_state.board = {}

    st.session_state.search_results = []

    st.session_state.interrogation = None

    st.session_state.prediction = None

    st.session_state.investigation = None

    st.session_state.fact_check = None

    st.session_state.verdict = None


# =========================================================
# API HELPERS
# =========================================================

def api_get(
    path,
    timeout=30,
):

    response = httpx.get(
        BACKEND_URL.rstrip("/")
        + path,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def api_post(
    path,
    payload,
    timeout=240,
):

    response = httpx.post(
        BACKEND_URL.rstrip("/")
        + path,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# HELPERS
# =========================================================

def safe(
    value
):

    return html.escape(
        str(
            value or ""
        )
    )


def short_text(
    text,
    length=340,
):

    text = str(
        text or ""
    )

    if len(text) <= length:

        return text

    return (
        text[:length]
        + "..."
    )


def show_citations(
    citations
):

    if not citations:

        st.caption(
            "No source notes were attached."
        )

        return


    for citation in citations:

        verified = citation.get(
            "source_verified",
            False,
        )


        source_title = (
            citation.get(
                "title"
            )
            or
            "Case evidence"
        )


        icon = (
            "✅"
            if verified
            else "⚠️"
        )


        with st.expander(
            (
                f"{icon} "
                f"{source_title}"
            )
        ):

            st.write(
                citation.get(
                    "claim",
                    ""
                )
            )


            excerpt = citation.get(
                "evidence_excerpt"
            )


            if excerpt:

                st.info(
                    excerpt
                )


            if not verified:

                st.warning(
                    (
                        "This source is unverified. "
                        "Treat it as a lead, not a fact."
                    )
                )


            with st.expander(
                "Case-file reference"
            ):

                st.code(
                    (
                        "Document: "
                        f"{citation.get('document_id')}\n"
                        "Evidence ID: "
                        f"{citation.get('chunk_id')}"
                    )
                )


def graph_to_dot(
    graph_data,
    max_nodes=55,
    max_edges=90,
):

    nodes = (
        graph_data.get(
            "nodes",
            []
        )[:max_nodes]
    )


    allowed_ids = {

        node.get("id")

        for node
        in nodes
    }


    lines = [
        "digraph G {",
        "rankdir=LR;",
        'graph [bgcolor="transparent"];',
        (
            'node [fontsize=10, '
            'fontcolor="#eadcc2", '
            'color="#8c6b4b"];'
        ),
        (
            'edge [fontsize=8, '
            'fontcolor="#bba586", '
            'color="#6f543d"];'
        ),
    ]


    for node in nodes:

        node_id = str(
            node.get(
                "id",
                ""
            )
        )


        label = str(
            node.get(
                "label",
                node_id
            )
        )


        if len(label) > 36:

            label = (
                label[:36]
                + "..."
            )


        label = label.replace(
            '"',
            '\\"'
        )


        node_type = node.get(
            "node_type"
        )


        shape = {
            "document": "box",
            "event": "diamond",
            "entity": "ellipse",
        }.get(
            node_type,
            "ellipse"
        )


        node_id_safe = node_id.replace(
            '"',
            '\\"'
        )


        lines.append(
            (
                f'"{node_id_safe}" '
                f'[label="{label}", '
                f'shape={shape}];'
            )
        )


    edge_count = 0


    for edge in graph_data.get(
        "edges",
        []
    ):

        if edge_count >= max_edges:

            break


        source = edge.get(
            "source"
        )

        target = edge.get(
            "target"
        )


        if (
            source not in allowed_ids
            or
            target not in allowed_ids
        ):

            continue


        label = str(
            edge.get(
                "label",
                ""
            )
        ).replace(
            '"',
            '\\"'
        )


        source_safe = str(
            source
        ).replace(
            '"',
            '\\"'
        )


        target_safe = str(
            target
        ).replace(
            '"',
            '\\"'
        )


        lines.append(
            (
                f'"{source_safe}" -> '
                f'"{target_safe}" '
                f'[label="{label}"];'
            )
        )


        edge_count += 1


    lines.append(
        "}"
    )


    return "\n".join(
        lines
    )


# =========================================================
# HEADER
# =========================================================

st.html(
    """
    <div class="detective-title">
        🕵️ The Casebook
    </div>

    <div class="detective-tagline">
        Every story leaves a trail.
        Your job is to follow it.
    </div>
    """
)


# =========================================================
# BACKEND CHECK
# =========================================================

try:

    api_get(
        "/health"
    )

except Exception:

    st.error(
        """
        The case archive is unavailable.

        Start the backend first:

        `python -m uvicorn backend.main:app --reload`
        """
    )

    st.stop()


# =========================================================
# LOAD CASES
# =========================================================

cases_response = (
    api_get(
        "/cases"
    )
)


cases = (
    cases_response.get(
        "cases",
        []
    )
)


if not cases:

    st.error(
        "The archive contains no cases."
    )

    st.stop()


case_map = {

    case["title"]:
        case["case_id"]

    for case
    in cases
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "🗂️ Your Case Board"
    )


    selected_title = (
        st.selectbox(
            "Open a case file",
            list(
                case_map.keys()
            ),
        )
    )


    selected_case_id = (
        case_map[
            selected_title
        ]
    )


    if (
        st.session_state.case_id
        != selected_case_id
    ):

        st.session_state.case_id = (
            selected_case_id
        )

        reset_case()


    st.divider()


    st.markdown(
        "### Case Progress"
    )


    st.write(
        "✅ Case file opened"
    )


    st.write(
        (
            "✅ Clues pinned"
            if st.session_state.board
            else
            "⬜ Find useful clues"
        )
    )


    st.write(
        (
            "✅ Hunch locked in"
            if st.session_state.prediction
            else
            "⬜ Make your first hunch"
        )
    )


    st.write(
        (
            "✅ Investigator consulted"
            if st.session_state.investigation
            else
            "⬜ Call in the Investigator"
        )
    )


    st.write(
        (
            "✅ Theory challenged"
            if st.session_state.fact_check
            else
            "⬜ Send in the Skeptic"
        )
    )


    st.write(
        (
            "✅ Case closed"
            if st.session_state.verdict
            else
            "⬜ Reach your verdict"
        )
    )


    with st.expander(
        "Behind the scenes"
    ):

        st.caption(
            (
                "Powered by hybrid retrieval, "
                "an evidence graph, an Investigator "
                "Agent and an adversarial Fact-Checker."
            )
        )


# =========================================================
# ACTIVE CASE
# =========================================================

case = api_get(
    (
        f"/cases/"
        f"{selected_case_id}"
    )
)


candidates = (
    case.get(
        "candidates",
        []
    )
)


candidate_map = {

    candidate["name"]:
        candidate

    for candidate
    in candidates
}


# =========================================================
# CASE INTRO
# =========================================================

st.html(
    f"""
    <div class="case-file">

        <div class="case-number">
            Confidential Case File
        </div>

        <h2>
            {safe(case['title'])}
        </h2>

        <div class="case-description">
            {safe(case.get('description', ''))}
        </div>

        <hr>

        <div class="mission-label">
            Your Assignment
        </div>

        <div class="mission-text">
            {safe(case['question'])}
        </div>

    </div>
    """
)


# =========================================================
# TABS
# =========================================================

(
    case_tab,
    investigate_tab,
    agents_tab,
    verdict_tab,
) = st.tabs([
    "📁 The Case File",
    "🔎 Detective Desk",
    "🤖 Consult the Agents",
    "⚖️ Close the Case",
])


# =========================================================
# TAB 1 — CASE FILE
# =========================================================

with case_tab:

    st.markdown(
        "### 👤 People & Groups of Interest"
    )


    st.write(
        (
            "These names keep showing up in the files. "
            "Some may be central to the case. "
            "Others may simply be part of the noise."
        )
    )


    column_count = max(
        1,
        min(
            len(candidates),
            4
        )
    )


    columns = (
        st.columns(
            column_count
        )
    )


    for index, candidate in enumerate(
        candidates
    ):

        with columns[
            index
            % column_count
        ]:

            st.html(
                f"""
                <div class="suspect-card">

                    <div class="suspect-name">
                        {safe(candidate['name'])}
                    </div>

                    <div class="suspect-type">
                        {safe(candidate['category'].title())}
                    </div>

                </div>
                """
            )


    st.divider()


    st.markdown(
        "### 🗃️ The Archive"
    )


    st.write(
        (
            "This case was pieced together from "
            f"**{len(case['document_ids'])} "
            "source documents**."
        )
    )


    if case.get(
        "unverified_evidence"
    ):

        st.warning(
            """
            📰 **Watch your step, Detective.**

            Not every lead in this file can be trusted.

            Somewhere in the case is an **unverified clue**.
            Check the source before building your theory around it.
            """
        )


    with st.expander(
        "Open the archive index"
    ):

        for document_id in (
            case["document_ids"]
        ):

            st.code(
                document_id
            )


# =========================================================
# TAB 2 — DETECTIVE DESK
# =========================================================

with investigate_tab:

    left, right = (
        st.columns(
            [1.3, 1]
        )
    )


    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    with left:

        st.markdown(
            "### 🔍 Search the Archives"
        )


        st.write(
            "Pick a lead and see where it takes you."
        )


        search_query = (
            st.text_input(
                "What clue are you hunting for?",
                placeholder=(
                    "Try: Taliban attacks, "
                    "prisoner handover, NATO..."
                ),
            )
        )


        if st.button(
            "🔎 Search the Files",
            type="primary",
        ):

            if not search_query.strip():

                st.warning(
                    "Give me a lead to follow first."
                )

            else:

                try:

                    with st.spinner(
                        "Searching through the files..."
                    ):

                        response = (
                            api_post(
                                "/search_evidence",
                                {
                                    "query":
                                        search_query,

                                    "case_id":
                                        selected_case_id,

                                    "limit":
                                        8,

                                    "mode":
                                        "hybrid",
                                },
                            )
                        )


                    st.session_state.search_results = (
                        response.get(
                            "results",
                            []
                        )
                    )

                except Exception as error:

                    st.error(
                        f"Search failed: {error}"
                    )


        for item in (
            st.session_state.search_results
        ):

            verified = (
                item.get(
                    "verified",
                    False
                )
            )


            reliability = (
                (
                    '<span class="verified-text">'
                    "✓ Verified source"
                    "</span>"
                )
                if verified
                else
                (
                    '<span class="unverified-text">'
                    "⚠ Unverified lead"
                    "</span>"
                )
            )


            st.html(
                f"""
                <div class="clue-card">

                    <div class="clue-title">
                        {safe(item.get('title', 'Case Clue'))}
                    </div>

                    <div>
                        {reliability}
                    </div>

                    <div class="clue-text">
                        {safe(short_text(item.get('text')))}
                    </div>

                </div>
                """
            )


            already_pinned = (
                item["chunk_id"]
                in
                st.session_state.board
            )


            if st.button(
                (
                    "✓ Clue pinned"
                    if already_pinned
                    else
                    "📌 Pin This Clue"
                ),
                key=(
                    "pin_"
                    + item["chunk_id"]
                ),
                disabled=already_pinned,
            ):

                st.session_state.board[
                    item["chunk_id"]
                ] = item

                st.rerun()


            with st.expander(
                "Read the full clipping"
            ):

                st.write(
                    item.get(
                        "text",
                        ""
                    )
                )


                st.caption(
                    (
                        "Archive reference: "
                        f"{item.get('document_id')}"
                    )
                )


    # -----------------------------------------------------
    # CANDIDATE DOSSIER
    # -----------------------------------------------------

    with right:

        st.markdown(
            "### 🗣️ Question the Dossier"
        )


        st.write(
            (
                "Choose a person or group and ask "
                "what the case files actually say about them."
            )
        )


        selected_candidate_name = (
            st.selectbox(
                "Whose file do you want to open?",
                list(
                    candidate_map.keys()
                ),
            )
        )


        candidate_question = (
            st.text_area(
                "What do you want to know?",
                placeholder=(
                    "What evidence connects them "
                    "to the incidents?"
                ),
            )
        )


        if st.button(
            "🗂️ Dig Into Their File"
        ):

            if not candidate_question.strip():

                st.warning(
                    "Ask the dossier a question first."
                )

            else:

                candidate = (
                    candidate_map[
                        selected_candidate_name
                    ]
                )


                try:

                    with st.spinner(
                        "Cross-checking the dossier..."
                    ):

                        st.session_state.interrogation = (
                            api_post(
                                "/interrogate",
                                {
                                    "case_id":
                                        selected_case_id,

                                    "candidate_id":
                                        candidate[
                                            "entity_id"
                                        ],

                                    "question":
                                        candidate_question,
                                },
                            )
                        )

                except Exception as error:

                    st.error(
                        f"Dossier search failed: {error}"
                    )


        interrogation = (
            st.session_state.interrogation
        )


        if interrogation:

            st.markdown(
                "#### 📜 What the Evidence Says"
            )


            st.write(
                interrogation.get(
                    "answer",
                    ""
                )
            )


            confidence = float(
                interrogation.get(
                    "confidence",
                    0
                )
            )


            st.progress(
                min(
                    max(
                        confidence,
                        0
                    ),
                    1
                )
            )


            st.caption(
                (
                    "Evidence confidence: "
                    f"{confidence:.0%}"
                )
            )


            if interrogation.get(
                "used_unverified_evidence"
            ):

                st.warning(
                    (
                        "Part of this answer comes "
                        "from an unverified lead."
                    )
                )


            show_citations(
                interrogation.get(
                    "citations",
                    []
                )
            )


    # -----------------------------------------------------
    # PINBOARD
    # -----------------------------------------------------

    st.divider()


    st.markdown(
        "### 📌 Your Pinboard"
    )


    if not st.session_state.board:

        st.info(
            (
                "Nothing pinned yet. "
                "Search the archives and save the clues "
                "that look important."
            )
        )


    else:

        for chunk_id, item in list(
            st.session_state.board.items()
        ):

            col1, col2 = (
                st.columns(
                    [6, 1]
                )
            )


            with col1:

                st.markdown(
                    (
                        f"**{item.get('title', 'Clue')}**"
                    )
                )


                st.write(
                    short_text(
                        item.get(
                            "text"
                        ),
                        240,
                    )
                )


                if not item.get(
                    "verified",
                    False
                ):

                    st.caption(
                        "⚠ Unverified lead"
                    )


            with col2:

                if st.button(
                    "Unpin",
                    key=(
                        "remove_"
                        + chunk_id
                    ),
                ):

                    del st.session_state.board[
                        chunk_id
                    ]

                    st.rerun()


    # -----------------------------------------------------
    # GRAPH
    # -----------------------------------------------------

    st.divider()


    with st.expander(
        "🕸️ Follow the Connections"
    ):

        st.write(
            (
                "Trace how people, groups, events "
                "and source documents connect."
            )
        )


        try:

            graph_data = api_get(
                (
                    f"/cases/"
                    f"{selected_case_id}"
                    "/graph"
                )
            )


            st.graphviz_chart(
                graph_to_dot(
                    graph_data
                ),
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                (
                    "Couldn't draw the connection board: "
                    f"{error}"
                )
            )


# =========================================================
# TAB 3 — AGENTS
# =========================================================

with agents_tab:

    # -----------------------------------------------------
    # USER HUNCH
    # -----------------------------------------------------

    st.markdown(
        "### 🧠 What's Your Hunch?"
    )


    st.write(
        """
        You've seen the files. You've followed a few leads.

        Before the AI detectives reveal their theory,
        put **your own hunch on record**.
        """
    )


    prediction_name = (
        st.selectbox(
            "Who stands out to you right now?",
            list(
                candidate_map.keys()
            ),
            key="prediction_name",
        )
    )


    prediction_reasoning = (
        st.text_area(
            "What's your reasoning?",
            placeholder=(
                "What made this candidate stand out?"
            ),
            key="prediction_reasoning",
        )
    )


    if st.button(
        "🔒 Lock In My Hunch"
    ):

        candidate = (
            candidate_map[
                prediction_name
            ]
        )


        try:

            result = api_post(
                "/predict",
                {
                    "case_id":
                        selected_case_id,

                    "candidate_id":
                        candidate[
                            "entity_id"
                        ],

                    "reasoning":
                        prediction_reasoning,
                },
            )


            st.session_state.prediction = (
                result[
                    "prediction"
                ]
            )


            st.success(
                (
                    "Your hunch is officially on the record."
                )
            )

        except Exception as error:

            st.error(
                f"Couldn't record your hunch: {error}"
            )


    if st.session_state.prediction:

        prediction = (
            st.session_state.prediction
        )


        st.info(
            (
                "🗒️ Your original hunch: "
                f"**{prediction['candidate_name']}**"
            )
        )


        st.divider()


        # -------------------------------------------------
        # INVESTIGATOR
        # -------------------------------------------------

        st.markdown(
            "### 🕵️ The Investigator"
        )


        st.write(
            """
            Time for a second pair of eyes.

            The Investigator will search the case,
            build a working theory, and chase another lead
            if the first evidence isn't convincing enough.
            """
        )


        if st.button(
            "🕵️ Call In the Investigator",
            type="primary",
        ):

            try:

                with st.spinner(
                    "The Investigator is working the case..."
                ):

                    st.session_state.investigation = (
                        api_post(
                            "/investigate",
                            {
                                "case_id":
                                    selected_case_id,

                                "question":
                                    case[
                                        "question"
                                    ],
                            },
                            timeout=300,
                        )
                    )


                    st.session_state.fact_check = None

            except Exception as error:

                st.error(
                    f"Investigator failed: {error}"
                )


        investigation = (
            st.session_state.investigation
        )


        if investigation:

            theory = (
                investigation.get(
                    "theory",
                    {}
                )
            )


            st.markdown(
                "#### 📜 The Investigator's Theory"
            )


            st.success(
                theory.get(
                    "statement",
                    ""
                )
            )


            col1, col2 = (
                st.columns(
                    2
                )
            )


            col1.metric(
                "Leading Candidate",
                theory.get(
                    "candidate_name",
                    "Undetermined"
                ),
            )


            confidence = float(
                theory.get(
                    "confidence",
                    0
                )
            )


            col2.metric(
                "Strength of Theory",
                f"{confidence:.0%}",
            )


            st.write(
                theory.get(
                    "rationale",
                    ""
                )
            )


            st.markdown(
                "##### Evidence behind the theory"
            )


            show_citations(
                investigation.get(
                    "citations",
                    []
                )
            )


            with st.expander(
                "🔎 How did the Investigator follow the trail?"
            ):

                for step in investigation.get(
                    "search_trace",
                    []
                ):

                    st.markdown(
                        (
                            f"**Lead "
                            f"{step['attempt']}**"
                        )
                    )


                    if step.get(
                        "purpose"
                    ):

                        st.write(
                            step[
                                "purpose"
                            ]
                        )


                    st.caption(
                        (
                            "Search used: "
                            f"{step['query']}"
                        )
                    )


            # ---------------------------------------------
            # SKEPTIC
            # ---------------------------------------------

            st.divider()


            st.markdown(
                "### 🧐 The Skeptic"
            )


            st.write(
                """
                Every good theory needs someone trying
                to tear it apart.

                The Skeptic deliberately hunts for
                contradictions, rival explanations and
                evidence the Investigator may have missed.
                """
            )


            if st.button(
                "🧐 Send In the Skeptic"
            ):

                try:

                    with st.spinner(
                        "Looking for cracks in the theory..."
                    ):

                        st.session_state.fact_check = (
                            api_post(
                                "/fact_check",
                                {
                                    "case_id":
                                        selected_case_id,

                                    "investigation":
                                        investigation,
                                },
                                timeout=300,
                            )
                        )

                except Exception as error:

                    st.error(
                        f"Skeptic failed: {error}"
                    )


        fact_check = (
            st.session_state.fact_check
        )


        if fact_check:

            st.markdown(
                "#### 🧪 The Skeptic's Report"
            )


            st.warning(
                fact_check.get(
                    "conclusion",
                    ""
                )
            )


            challenge_strength = float(
                fact_check.get(
                    "challenge_strength",
                    0
                )
            )


            st.metric(
                "How badly was the theory shaken?",
                f"{challenge_strength:.0%}",
            )


            for finding in fact_check.get(
                "findings",
                []
            ):

                title = (
                    finding.get(
                        "finding_type",
                        "Finding"
                    )
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                )


                with st.expander(
                    f"🔎 {title}"
                ):

                    st.write(
                        finding.get(
                            "statement",
                            ""
                        )
                    )


                    show_citations(
                        finding.get(
                            "citations",
                            []
                        )
                    )


            with st.expander(
                "See how the Skeptic searched"
            ):

                for step in fact_check.get(
                    "search_trace",
                    []
                ):

                    st.markdown(
                        (
                            f"**Counter-lead "
                            f"{step['attempt']}**"
                        )
                    )


                    if step.get(
                        "purpose"
                    ):

                        st.write(
                            step[
                                "purpose"
                            ]
                        )


                    st.caption(
                        step[
                            "query"
                        ]
                    )


    else:

        st.warning(
            """
            🔒 The AI detectives aren't showing
            their cards yet.

            Lock in your own hunch first.
            """
        )


# =========================================================
# TAB 4 — FINAL VERDICT
# =========================================================

with verdict_tab:

    st.markdown(
        "### ⚖️ Time to Close the Case"
    )


    st.write(
        """
        You've searched the archive.

        You've opened the dossiers.

        You've heard the Investigator and the Skeptic.

        Now put **your final conclusion on the table**.
        """
    )


    final_candidate_name = (
        st.selectbox(
            "Who does the evidence point to?",
            list(
                candidate_map.keys()
            ),
            key="final_candidate",
        )
    )


    final_reasoning = (
        st.text_area(
            "Walk us through your reasoning",
            placeholder=(
                "Which clues mattered most? "
                "What convinced you?"
            ),
        )
    )


    evidence_options = {

        (
            f"{item.get('title', 'Clue')} — "
            f"{short_text(item.get('text'), 75)}"
        ):
            chunk_id

        for chunk_id, item
        in st.session_state.board.items()
    }


    selected_labels = (
        st.multiselect(
            "Which clues back your conclusion?",
            list(
                evidence_options.keys()
            ),
        )
    )


    selected_chunk_ids = [

        evidence_options[
            label
        ]

        for label
        in selected_labels
    ]


    if not st.session_state.board:

        st.info(
            """
            Your pinboard is empty.

            Head back to the Detective Desk and pin
            a few clues before closing the case.
            """
        )


    if st.button(
        "🔐 Close the Case",
        type="primary",
    ):

        candidate = (
            candidate_map[
                final_candidate_name
            ]
        )


        try:

            with st.spinner(
                "Reviewing your case..."
            ):

                st.session_state.verdict = (
                    api_post(
                        "/submit_verdict",
                        {
                            "case_id":
                                selected_case_id,

                            "candidate_id":
                                candidate[
                                    "entity_id"
                                ],

                            "reasoning":
                                final_reasoning,

                            "evidence_chunk_ids":
                                selected_chunk_ids,
                        },
                        timeout=300,
                    )
                )

        except Exception as error:

            st.error(
                f"Couldn't close the case: {error}"
            )


    verdict = (
        st.session_state.verdict
    )


    if verdict:

        st.divider()


        st.markdown(
            "### 📜 Case Review"
        )


        status = (
            verdict.get(
                "status",
                ""
            )
            .replace(
                "_",
                " "
            )
            .title()
        )


        score = float(
            verdict.get(
                "score",
                0
            )
        )


        col1, col2 = (
            st.columns(
                2
            )
        )


        col1.metric(
            "How Strong Is Your Case?",
            status,
        )


        col2.metric(
            "Evidence Strength",
            f"{score:.0%}",
        )


        st.progress(
            min(
                max(
                    score,
                    0
                ),
                1
            )
        )


        st.write(
            verdict.get(
                "explanation",
                ""
            )
        )


        st.markdown(
            "#### Clues that held up"
        )


        show_citations(
            verdict.get(
                "supporting_citations",
                []
            )
        )


        st.success(
            """
            🕵️ **Case closed.**

            Whether your first hunch survived or not,
            you followed the evidence — exactly as
            a good detective should.
            """
        )
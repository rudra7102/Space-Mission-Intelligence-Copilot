// ── Configure marked ──────────────────────────
marked.setOptions({ breaks: true, gfm: true });

// ── DOM refs ──────────────────────────────────
const form        = document.getElementById('chatForm');
const input       = document.getElementById('userInput');
const btn         = document.getElementById('sendBtn');
const chatWrapper = document.getElementById('chatWrapper');

// ── Helpers ───────────────────────────────────
function scrollToBottom() {
    chatWrapper.scrollTop = chatWrapper.scrollHeight;
}

function showTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message system typing-indicator';
    el.id = 'typingIndicator';
    el.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-bubble" style="display:flex;align-items:center;gap:4px;">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>`;
    chatWrapper.appendChild(el);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

// ── Build collapsible tool trace HTML ─────────
function buildToolTrace(toolCalls) {
    if (!toolCalls || toolCalls.length === 0) return '';
    const t = toolCalls[0];
    const iconMap = {
        SearchKB:          'fa-magnifying-glass',
        ComputeSuccessRate:'fa-chart-line',
        GetLaunchWindow:   'fa-rocket',
        GetPolicy:         'fa-scroll',
        CreateTicket:      'fa-triangle-exclamation',
    };
    const icon = iconMap[t.tool] || 'fa-microchip';
    const prettyIn  = JSON.stringify(t.input,  null, 2);
    const prettyOut = JSON.stringify(t.output, null, 2);
    const outTrunc  = prettyOut.length > 600 ? prettyOut.slice(0, 600) + '\n...' : prettyOut;

    return `
    <div class="tool-trace-wrap">
        <button class="tool-toggle" onclick="
            this.nextElementSibling.classList.toggle('open');
            this.querySelector('.chevron').classList.toggle('rotate');">
            <i class="fa-solid ${icon}"></i>
            <span>Tool Used: <strong>${t.tool}</strong></span>
            <i class="fa-solid fa-chevron-down chevron"></i>
        </button>
        <div class="tool-detail">
            <div class="tool-section-label">Input</div>
            <pre>${prettyIn}</pre>
            <div class="tool-section-label" style="margin-top:8px;">Output</div>
            <pre>${outTrunc}</pre>
        </div>
    </div>`;
}

// ── Build citations HTML ───────────────────────
function buildCitations(citations) {
    if (!citations || citations.length === 0) return '';
    let html = `<div class="citations-box"><strong>📎 Sources</strong><ul>`;
    citations.slice(0, 3).forEach(c => {
        const snippet = c.passage ? c.passage.slice(0, 120) + '…' : '';
        html += `<li><code>[${c.doc_id.slice(0, 12)}]</code> ${snippet}</li>`;
    });
    return html + `</ul></div>`;
}

// ── Build escalation badge HTML ────────────────
function buildEscalation(escalated, ticketId) {
    if (!escalated) return '';
    return `<div class="escalate-badge">
        <i class="fa-solid fa-triangle-exclamation"></i>
        Escalated — Ticket <code>${ticketId}</code>
    </div>`;
}

// ── Build confidence bar HTML ──────────────────
function buildConfidenceBar(confidence) {
    if (confidence === undefined || confidence === null) return '';
    const pct = Math.round(confidence * 100);
    return `<div class="confidence-bar-wrap">
        <span class="confidence-label">Confidence</span>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:${pct}%;"></div>
        </div>
        <span class="confidence-pct">${pct}%</span>
    </div>`;
}

// ── Main addMessage ────────────────────────────
function addMessage(type, content, apiData) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;
    const icon = type === 'user' ? 'fa-user' : 'fa-robot';

    const parsedContent = type === 'system'
        ? `<div class="markdown-body">${marked.parse(content)}</div>`
        : `<p>${content}</p>`;

    let extras = '';
    if (type === 'system' && apiData && typeof apiData === 'object') {
        extras += buildToolTrace(apiData.tool_calls);
        extras += buildCitations(apiData.citations);
        
        // ── Google search link for KB-gap queries ──
        if (apiData.google_url) {
            extras += `<div class="google-search-link">
                <a href="${apiData.google_url}" target="_blank" rel="noopener noreferrer">
                    <i class="fa-solid fa-magnifying-glass"></i>
                    Search on Google (NASA / ESA / SpaceX)
                    <i class="fa-solid fa-arrow-up-right-from-square" style="font-size:11px;margin-left:6px;"></i>
                </a>
            </div>`;
        }
        
        extras += buildEscalation(apiData.escalated, apiData.ticket_id);
        extras += buildConfidenceBar(apiData.confidence);
    }

    msgDiv.innerHTML = `
        <div class="avatar"><i class="fa-solid ${icon}"></i></div>
        <div class="msg-bubble">
            ${parsedContent}
            ${extras}
        </div>`;

    chatWrapper.appendChild(msgDiv);
    scrollToBottom();
}

// ── Form submit handler ────────────────────────
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    addMessage('user', query, null);
    input.value = '';
    btn.disabled = true;
    showTypingIndicator();

    try {
        const res = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query })
        });

        if (!res.ok) throw new Error(`Server responded ${res.status}`);
        const data = await res.json();
        removeTypingIndicator();
        addMessage('system', data.answer, data);

    } catch (err) {
        removeTypingIndicator();
        addMessage('system', '⚠️ Error communicating with Mission Control. Please try again.', null);
        console.error(err);
    } finally {
        btn.disabled = false;
        input.focus();
    }
});

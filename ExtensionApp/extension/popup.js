// =====================================================================
// BEC Shield — popup.js
// =====================================================================

// ---- Scrape email content from the active tab ----
function scrapeEmailContent() {
    const host = window.location.hostname.toLowerCase();

    const looksLikeEmailInbox = (
        host.includes('mail.google.com') ||
        host.includes('outlook.office.com') ||
        host.includes('outlook.live.com') ||
        host.includes('mail.yahoo.com')
    );

    if (!looksLikeEmailInbox) {
        return { ok: false, reason: 'not_email_inbox' };
    }

    const gmailBody = document.querySelector('.a3s.aiL');
    const outlookBody = document.querySelector('[aria-label="Message body"]');
    const emailBody = gmailBody || outlookBody;

    if (!emailBody) {
        return { ok: false, reason: 'not_open_email' };
    }

    const subject = document.title || "";

    function normalizeSenderLabel(nameText, emailText) {
        const cleanedName = (nameText || '').trim();
        const cleanedEmail = (emailText || '').trim();
        if (cleanedName && cleanedName.toLowerCase() !== 'me') return cleanedName;
        if (cleanedEmail) return cleanedEmail;
        return "Unknown Sender";
    }

    let senderName = "";
    let senderEmail = "";

    const gmailSenderName = document.querySelector('span[email]');
    if (gmailSenderName) {
        senderName = (gmailSenderName.textContent || "").trim();
        senderEmail = (gmailSenderName.getAttribute('email') || "").trim();
        if (senderName.toLowerCase() === 'me' && senderEmail) senderName = "";
    }

    const outlookFromField = document.querySelector('[data-test-id="from-field"]');
    if (outlookFromField && !senderName) {
        senderName = outlookFromField.textContent || "";
    }

    if (!senderName) {
        const headerElements = document.querySelectorAll('[role="heading"], .bAk, [data-tooltip-id]');
        for (let elem of headerElements) {
            const text = elem.textContent;
            if (text && text.includes('@')) { senderEmail = text; break; }
        }
    }

    const displaySender = normalizeSenderLabel(senderName, senderEmail);
    const headerText = `From: ${displaySender}\nSubject: ${subject}`;
    const bodyText = emailBody.innerText || emailBody.textContent || "";
    let combinedText = headerText + "\n\n" + bodyText;

    return {
        ok: true,
        text: combinedText.substring(0, 10000),
        sender: displaySender,
        subject: subject,
        bodyText: bodyText.substring(0, 8000)
    };
}

// =====================================================================
// State / helpers
// =====================================================================
const MANUAL_DRAFT_KEY = 'phishing_detector_manual_draft';

function getSelectedModel() {
    return 'rf';
}

function getManualInputs() {
    return {
        subject: document.getElementById('inputSubject'),
        header: document.getElementById('inputHeader'),
        body: document.getElementById('inputBody'),
    };
}

function buildManualText() {
    const { subject, header, body } = getManualInputs();
    const subjectText = subject.value.trim();
    const headerText  = header.value.trim();
    const bodyText    = body.value.trim();
    return [
        subjectText ? `Subject: ${subjectText}` : '',
        headerText  ? `${headerText}` : '',
        bodyText    ? `${bodyText}` : ''
    ].filter(Boolean).join('\n\n').trim();
}

function saveManualDraft() {
    const { subject, header, body } = getManualInputs();
    chrome.storage.local.set({
        [MANUAL_DRAFT_KEY]: {
            subject: subject.value,
            header: header.value,
            body: body.value,
        }
    });
}

function restoreManualDraft() {
    const { subject, header, body } = getManualInputs();
    chrome.storage.local.get([MANUAL_DRAFT_KEY], (result) => {
        const draft = result[MANUAL_DRAFT_KEY];
        if (!draft) return;
        subject.value = draft.subject || '';
        header.value  = draft.header  || '';
        body.value    = draft.body    || '';
    });
}

function initPopupPersistence() {
    restoreManualDraft();
    const { subject, header, body } = getManualInputs();
    [subject, header, body].forEach((input) => {
        input.addEventListener('input', saveManualDraft);
        input.addEventListener('change', saveManualDraft);
        input.addEventListener('blur', saveManualDraft);
        input.addEventListener('paste', () => window.setTimeout(saveManualDraft, 0));
    });
    window.addEventListener('beforeunload', saveManualDraft);
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') saveManualDraft();
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPopupPersistence);
} else {
    initPopupPersistence();
}

// =====================================================================
// Result rendering
// =====================================================================

/**
 * Determine if a reason is "bad" (phishing signal) or "good" (safe signal).
 * Returns 'bad', 'good', or 'neutral'.
 */
function classifyReason(text) {
    const lower = text.toLowerCase();
    const badKeywords = [
        'differs', 'mismatch', 'never been seen', 'first-time', 'free public email',
        'spoofed', 'impersonation', 'attacker', 'flagged', 'highly similar to known',
        'phishing probability', 'wire-transfer', 'urgency', 'elevated', 'suspicious',
        'moderate phishing', 'high risk', 'identity spoofing'
    ];
    const goodKeywords = [
        'matches', 'consistent', 'corporate or custom', 'frequently observed',
        'well-known', 'legitimate', 'low impersonation', 'passed the safety gate',
        'no name spoofing', 'not match known phishing'
    ];
    if (badKeywords.some(k => lower.includes(k))) return 'bad';
    if (goodKeywords.some(k => lower.includes(k))) return 'good';
    return 'neutral';
}

function formatStageName(key) {
    const map = {
        'stage_1_impersonation_probability': 'Stage 1 · Header Gate',
        'stage_2_body_phishing_probability': 'Stage 2 · Body Analysis',
        'final_stacked_probability':         'Final · Stacked Decision',
    };
    return map[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function probClass(val) {
    if (val >= 0.7) return 'high';
    if (val >= 0.4) return 'med';
    return 'low';
}

function renderResult(data) {
    const isPhishing = data.prediction === 'Phishing';
    const confidence = data.confidence !== undefined ? data.confidence : (data.bec_probability || 0);
    const pct = Math.round(confidence * 100);

    const explanation = data.explanation || {};
    const summary = explanation.summary || '';
    const reasons = explanation.reasons || [];
    const breakdown = explanation.stage_breakdown || {};

    // Show the card
    const resultBox = document.getElementById('resultBox');
    resultBox.classList.remove('hidden');

    // ---- Verdict Banner ----
    const banner = document.getElementById('verdictBanner');
    banner.className = `verdict-banner ${isPhishing ? 'phishing' : 'safe'}`;

    document.getElementById('verdictIcon').textContent = isPhishing ? '🚨' : '✅';

    const titleEl = document.getElementById('verdictTitle');
    titleEl.textContent = isPhishing ? 'Phishing Detected' : 'Email Appears Safe';
    titleEl.className = `verdict-title ${isPhishing ? 'phishing' : 'safe'}`;

    document.getElementById('verdictSubtitle').textContent =
        isPhishing
            ? 'This email shows signs of Business Email Compromise.'
            : 'No significant threat indicators were detected.';

    // ---- Probability Meter ----
    const probVal = document.getElementById('probValue');
    probVal.textContent = `${pct}%`;
    probVal.className = `prob-value ${isPhishing ? 'phishing' : 'safe'}`;

    const probBar = document.getElementById('probBar');
    probBar.className = `prob-bar ${isPhishing ? 'phishing' : 'safe'}`;
    // Animate after frame
    requestAnimationFrame(() => {
        probBar.style.width = `${pct}%`;
    });

    // ---- Summary ----
    const summarySection = document.getElementById('summarySection');
    const summaryText = document.getElementById('summaryText');
    if (summary) {
        summaryText.textContent = summary;
        summarySection.classList.remove('hidden');
    } else {
        summarySection.classList.add('hidden');
    }

    // ---- Reasons ----
    const reasonsSection = document.getElementById('reasonsSection');
    const reasonsList = document.getElementById('reasonsList');
    reasonsList.innerHTML = '';

    if (reasons.length > 0) {
        reasons.forEach(text => {
            const cls = classifyReason(text);
            const icon = cls === 'bad' ? '⚠️' : cls === 'good' ? '✓' : '•';
            const li = document.createElement('li');
            li.className = `reason-${cls !== 'neutral' ? cls : 'good'}`;
            li.innerHTML = `<span class="reason-icon">${icon}</span><span>${text}</span>`;
            reasonsList.appendChild(li);
        });
        reasonsSection.classList.remove('hidden');
    } else {
        reasonsSection.classList.add('hidden');
    }

    // ---- Stage Breakdown ----
    const breakdownSection = document.getElementById('breakdownSection');
    const breakdownGrid = document.getElementById('breakdownGrid');
    breakdownGrid.innerHTML = '';

    const breakdownKeys = Object.keys(breakdown);
    if (breakdownKeys.length > 0) {
        breakdownKeys.forEach(key => {
            const val = breakdown[key];
            const pctStr = (val * 100).toFixed(1) + '%';
            const cls = probClass(val);
            const item = document.createElement('div');
            item.className = 'breakdown-item';
            item.innerHTML = `
                <span class="breakdown-key">${formatStageName(key)}</span>
                <span class="breakdown-val ${cls}">${pctStr}</span>
            `;
            breakdownGrid.appendChild(item);
        });
        breakdownSection.classList.remove('hidden');
    } else {
        breakdownSection.classList.add('hidden');
    }
}

// =====================================================================
// Toggle Reasons visibility
// =====================================================================
document.getElementById('toggleReasons').addEventListener('click', () => {
    const list = document.getElementById('reasonsList');
    const btn  = document.getElementById('toggleReasons');
    const isHidden = list.classList.toggle('hidden');
    btn.textContent = isHidden ? 'Show' : 'Hide';
});

// =====================================================================
// Scan Page Button
// =====================================================================
document.getElementById('scanPageBtn').addEventListener('click', async () => {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        const loader    = document.getElementById('loader');
        const resultBox = document.getElementById('resultBox');
        loader.classList.remove('hidden');
        resultBox.classList.add('hidden');

        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: scrapeEmailContent,
        }, async (injectionResults) => {
            if (chrome.runtime.lastError || !injectionResults || !injectionResults.length) {
                alert('Cannot read this page. Try pasting text manually.');
                loader.classList.add('hidden');
                return;
            }

            const scanResult = injectionResults[0].result;
            if (!scanResult || scanResult.ok === false) {
                alert('Not an email inbox. Open an email and try again, or paste manually below.');
                loader.classList.add('hidden');
                return;
            }

            const scrapedText    = scanResult.text;
            const scrapedSender  = scanResult.sender || "Unknown";
            const scrapedSubject = scanResult.subject || "";
            const scrapedBody    = scanResult.bodyText || "";

            if (scrapedText && scrapedText.trim().length > 0) {
                const { subject, header, body } = getManualInputs();
                subject.value = scrapedSubject;
                header.value  = `From: ${scrapedSender}`;
                body.value    = scrapedBody;
                saveManualDraft();

                try {
                    const response = await fetch('http://127.0.0.1:5000/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: scrapedText, model: getSelectedModel() })
                    });
                    const data = await response.json();
                    if (response.ok) {
                        renderResult(data);
                    } else {
                        alert(data.error || 'Failed to analyze text.');
                    }
                } catch (error) {
                    console.error('Fetch error:', error);
                    alert('Could not connect to the backend server. Make sure the Flask API is running on port 5000.');
                } finally {
                    loader.classList.add('hidden');
                }
            } else {
                alert('No content found on this page to scan.');
                loader.classList.add('hidden');
            }
        });
    } catch (err) {
        console.error('Script injection error:', err);
        alert('Could not access the page content.');
    }
});

// =====================================================================
// Analyze Button (manual)
// =====================================================================
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = buildManualText();

    if (!text) {
        alert('Please enter at least one of Subject, Header, or Body.');
        return;
    }

    const loader    = document.getElementById('loader');
    const resultBox = document.getElementById('resultBox');
    loader.classList.remove('hidden');
    resultBox.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, model: getSelectedModel() })
        });
        const data = await response.json();

        if (response.ok) {
            renderResult(data);
        } else {
            alert(data.error || 'Failed to analyze text.');
        }
    } catch (error) {
        console.error('Fetch error:', error);
        alert('Could not connect to the backend server. Make sure the Flask API is running on port 5000.');
    } finally {
        loader.classList.add('hidden');
    }
});

// =====================================================================
// Reset Button
// =====================================================================
document.getElementById('resetBtn').addEventListener('click', () => {
    const { subject, header, body } = getManualInputs();
    subject.value = '';
    header.value  = '';
    body.value    = '';
    document.getElementById('resultBox').classList.add('hidden');
    saveManualDraft();
});

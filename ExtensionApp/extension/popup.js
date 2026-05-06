// Function to be injected into the active web page
function scrapeEmailContent() {
    // Grab the page title, which often contains the email subject
    let subject = document.title || "";
    let bodyText = "";

    // 1. Try to find Gmail's email body
    const gmailBody = document.querySelector('.a3s.aiL');
    // 2. Try to find Outlook's email body (generic selector)
    const outlookBody = document.querySelector('[aria-label="Message body"]');
    // 3. Fallback: Any selected text on the page
    const selection = window.getSelection().toString();

    if (selection) {
        bodyText = selection;
    } else if (gmailBody) {
        bodyText = gmailBody.innerText;
    } else if (outlookBody) {
        bodyText = outlookBody.innerText;
    } else {
        // 4. Last resort: Overall scan. Get all visible text on the page.
        bodyText = document.body.innerText;
    }

    // Combine subject and body, limit to 5000 characters to prevent overloading the model
    let combinedText = "Subject: " + subject + "\n\n" + bodyText;
    return combinedText.substring(0, 5000);
}

// Handle Auto-Scan button
document.getElementById('scanPageBtn').addEventListener('click', async () => {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        // Show loading state immediately for a seamless feel
        const loader = document.getElementById('loader');
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

            const scrapedText = injectionResults[0].result;
            if (scrapedText && scrapedText.trim().length > 0) {
                
                // Do the analysis blindly in the background without populating the text box
                try {
                    const response = await fetch('http://127.0.0.1:5000/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: scrapedText })
                    });

                    const data = await response.json();
                    const resultStatus = document.getElementById('resultStatus');
                    const confidenceLevel = document.getElementById('confidenceLevel');

                    if (response.ok) {
                        resultStatus.textContent = `Result: ${data.prediction}`;
                        resultStatus.style.color = data.prediction === 'Phishing' ? '#d32f2f' : '#388e3c';
                        confidenceLevel.textContent = `Phishing Probability: ${(data.confidence * 100).toFixed(1)}%`;
                        resultBox.classList.remove('hidden');
                    } else {
                        alert(data.error || 'Failed to analyze text.');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Could not connect to the backend server. Make sure the Flask api is running.');
                } finally {
                    loader.classList.add('hidden');
                }

            } else {
                alert('No content found on this page to scan.');
                loader.classList.add('hidden');
            }
        });
    } catch (err) {
        console.error('Error injecting script:', err);
        alert('Could not access the page content.');
    }
});

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = document.getElementById('inputText').value.trim();
    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }

    const loader = document.getElementById('loader');
    const resultBox = document.getElementById('resultBox');
    const resultStatus = document.getElementById('resultStatus');
    const confidenceLevel = document.getElementById('confidenceLevel');

    // Reset UI
    loader.classList.remove('hidden');
    resultBox.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();
        
        if (response.ok) {
            resultStatus.textContent = `Result: ${data.prediction}`;
            resultStatus.style.color = data.prediction === 'Phishing' ? '#d32f2f' : '#388e3c';
            confidenceLevel.textContent = `Phishing Probability: ${(data.confidence * 100).toFixed(1)}%`;
            resultBox.classList.remove('hidden');
        } else {
            alert(data.error || 'Failed to analyze text.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Could not connect to the backend server. Make sure the Flask api is running.');
    } finally {
        loader.classList.add('hidden');
    }
});

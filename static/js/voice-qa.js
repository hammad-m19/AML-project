/* Voice QA using Web Speech API (SpeechRecognition + SpeechSynthesis) */

(function () {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let lastAnswer = '';

  const questionInput = document.getElementById('qa-question');
  const startBtn = document.getElementById('start-listening');
  const stopBtn = document.getElementById('stop-listening');
  const voiceStatus = document.getElementById('voice-status');
  const askBtn = document.getElementById('ask-btn');
  const speakAnswerBtn = document.getElementById('speak-answer');
  const resultsPanel = document.getElementById('qa-results');
  const answerText = document.getElementById('qa-answer-text');
  const confidenceEl = document.getElementById('qa-confidence');

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      voiceStatus.textContent = 'Listening... speak your question';
      startBtn.disabled = true;
      stopBtn.disabled = false;
    };

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      questionInput.value = transcript;
    };

    recognition.onend = () => {
      voiceStatus.textContent = 'Voice capture complete. Click Get Answer or speak again.';
      startBtn.disabled = false;
      stopBtn.disabled = true;
      if (questionInput.value.trim()) {
        askQuestion(true);
      }
    };

    recognition.onerror = (event) => {
      voiceStatus.textContent = `Voice error: ${event.error}`;
      startBtn.disabled = false;
      stopBtn.disabled = true;
    };

    startBtn.addEventListener('click', () => recognition.start());
    stopBtn.addEventListener('click', () => recognition.stop());
  } else {
    startBtn.disabled = true;
    voiceStatus.textContent = 'Voice input not supported in this browser. Use Chrome or Edge.';
  }

  document.querySelectorAll('.sample-q').forEach(btn => {
    btn.addEventListener('click', () => {
      questionInput.value = btn.dataset.q;
    });
  });

  const loadSampleBtn = document.getElementById('load-sample-context');
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener('click', async () => {
      try {
        const contextPanel = document.getElementById('context-panel');
        if (contextPanel && !contextPanel.open) contextPanel.open = true;
        const res = await fetch('/sample/qa_context.txt');
        document.getElementById('qa-context').value = await res.text();
      } catch (e) {
        alert('Could not load sample context.');
      }
    });
  }

  askBtn.addEventListener('click', () => askQuestion(false));

  async function askQuestion(autoSpeak) {
    const contextEl = document.getElementById('qa-context');
    const context = contextEl ? contextEl.value.trim() : '';
    const question = questionInput.value.trim();
    if (!question) {
      alert('Please enter or speak a question.');
      return;
    }

    resultsPanel.style.display = 'block';
    answerText.textContent = 'Thinking...';
    confidenceEl.textContent = '';

    try {
      const res = await fetch('/api/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context, question })
      });
      const data = await res.json();
      if (data.error) {
        answerText.textContent = data.error;
        return;
      }
      lastAnswer = data.answer;
      answerText.textContent = data.answer;

      // Show mode (open-ended vs contextual) and model info
      const modeLabel = data.mode === 'contextual' ? '📄 Contextual answer' : '🌐 Open-ended answer';
      confidenceEl.textContent = `${modeLabel} · Model: ${data.model || 'Flan-T5'}`;

      if (autoSpeak) speakText(data.answer);
    } catch (e) {
      answerText.textContent = e.message;
    }
  }

  speakAnswerBtn.addEventListener('click', () => {
    if (lastAnswer) speakText(lastAnswer);
  });

  function speakText(text) {
    if (!window.speechSynthesis) {
      alert('Speech synthesis not supported in this browser.');
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }
})();

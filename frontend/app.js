/**
 * NovaTech Support AI — Customer Support Dashboard
 * Category-aware RAG frontend with sidebar topic navigation.
 * Pure Vanilla JavaScript • Zero external dependencies.
 */

(function () {
  'use strict';

  // -------------------------------------------------------------------------
  // API Configuration
  // -------------------------------------------------------------------------
  const BASE_API_URL      = '';
  const CHAT_ENDPOINT     = `${BASE_API_URL}/chat`;
  const KB_STATUS_ENDPOINT = `${BASE_API_URL}/knowledge-base/status`;

  // -------------------------------------------------------------------------
  // Topic Configuration
  // Maps topic key → display info + suggested questions
  // -------------------------------------------------------------------------
  const TOPICS = {
    payment_faq: {
      label: 'Payment FAQ',
      description: 'Get answers about accepted payment methods, billing cycles, invoice disputes, and payment security.',
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
               <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>
             </svg>`,
      suggestions: [
        'What payment methods does NovaTech accept?',
        'Can I pay with cryptocurrency?',
        'How do I get an invoice for my order?',
        'Is my payment information secure?'
      ]
    },
    troubleshooting: {
      label: 'Troubleshooting',
      description: 'Resolve technical issues with your NovaTech device — connectivity, performance, software, and hardware problems.',
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
               <circle cx="12" cy="12" r="3"/>
               <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
             </svg>`,
      suggestions: [
        'My device won\'t turn on — what should I do?',
        'How do I factory reset my NovaTech device?',
        'My NovaTech won\'t connect to Wi-Fi',
        'How do I update the firmware on my device?'
      ]
    },
    returns_refunds: {
      label: 'Returns & Refunds',
      description: 'Understand the return window, eligibility conditions, restocking fees, and how refunds are processed.',
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
               <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.49"/>
             </svg>`,
      suggestions: [
        'What is the standard return window?',
        'Can I return an opened item?',
        'How long does a refund take to process?',
        'Who pays for return shipping?'
      ]
    },
    shipping: {
      label: 'Shipping',
      description: 'Ask questions about shipping speeds, costs, international delivery, and order tracking.',
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
               <rect x="1" y="3" width="15" height="13"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/>
               <circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>
             </svg>`,
      suggestions: [
        'How long does standard shipping take?',
        'What does express shipping cost?',
        'Do you ship internationally?',
        'How do I track my order?'
      ]
    },
    warranty: {
      label: 'Warranty',
      description: 'Ask questions about product warranty coverage, duration, conditions, and how to submit a warranty claim.',
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
               <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
             </svg>`,
      suggestions: [
        'How long is the warranty on NovaTech products?',
        'What does the warranty cover?',
        'How do I make a warranty claim?',
        'Is accidental damage covered under warranty?'
      ]
    },
    nova_products: {
      label: 'Nova Products',
      description: 'Explore NovaTech\'s product lineup — specifications, features, compatibility, and accessories.',
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
               <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
             </svg>`,
      suggestions: [
        'What laptops does NovaTech offer?',
        'What is the battery life of the NovaBook Pro?',
        'What are the specs of the NovaPhone flagship?',
        'Does NovaTech have wireless earbuds?'
      ]
    }
  };

  // -------------------------------------------------------------------------
  // Application State
  // -------------------------------------------------------------------------
  const state = {
    selectedTopic: null,   // null = "all topics" / unrestricted
    conversationHistory: [],
    isGenerating: false,
    hasStartedChat: false,
    sidebarOpen: false
  };

  // -------------------------------------------------------------------------
  // DOM References
  // -------------------------------------------------------------------------
  const sidebar         = document.getElementById('sidebar');
  const sidebarOverlay  = document.getElementById('sidebarOverlay');
  const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
  const sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
  const sidebarNewChatBtn = document.getElementById('sidebarNewChatBtn');
  const topicNav        = document.getElementById('topicNav');
  const welcomeState    = document.getElementById('welcomeState');
  const welcomeTopicGrid = document.getElementById('welcomeTopicGrid');
  const topicContextCard = document.getElementById('topicContextCard');
  const tccTitle        = document.getElementById('tccTitle');
  const tccDesc         = document.getElementById('tccDesc');
  const tccIcon         = document.getElementById('tccIcon');
  const tccChips        = document.getElementById('tccChips');
  const headerTopicLabel = document.getElementById('headerTopicLabel');
  const chatMain        = document.getElementById('chatMain');
  const chatFeed        = document.getElementById('chatFeed');
  const chatForm        = document.getElementById('chatForm');
  const chatInput       = document.getElementById('chatInput');
  const sendBtn         = document.getElementById('sendBtn');
  const clearChatBtn    = document.getElementById('clearChatBtn');
  const charCounter     = document.getElementById('charCounter');
  const inputTopicBadge = document.getElementById('inputTopicBadge');
  const statusLabel     = document.getElementById('statusLabel');
  const systemStatusChip = document.getElementById('systemStatusChip');

  // -------------------------------------------------------------------------
  // Initialise
  // -------------------------------------------------------------------------
  function init() {
    setupEventListeners();
    checkSystemStatus();
    adjustTextareaHeight();
  }

  // -------------------------------------------------------------------------
  // Event Listeners
  // -------------------------------------------------------------------------
  function setupEventListeners() {
    // Chat form
    chatForm.addEventListener('submit', handleFormSubmit);
    chatInput.addEventListener('input', handleInputChange);
    chatInput.addEventListener('keydown', handleKeydown);

    // Sidebar navigation topic clicks (delegated)
    topicNav.addEventListener('click', (e) => {
      const btn = e.target.closest('.topic-item[data-topic]');
      if (btn) selectTopic(btn.dataset.topic);
    });

    // Welcome screen topic grid clicks (delegated)
    if (welcomeTopicGrid) {
      welcomeTopicGrid.addEventListener('click', (e) => {
        const card = e.target.closest('.welcome-topic-card[data-topic]');
        if (card) selectTopic(card.dataset.topic);
      });
    }

    // Topic context card — suggested question chips (delegated)
    tccChips.addEventListener('click', (e) => {
      const chip = e.target.closest('.suggestion-chip');
      if (chip && !state.isGenerating) {
        const prompt = chip.dataset.prompt;
        if (prompt) sendMessage(prompt);
      }
    });

    // New Chat buttons
    clearChatBtn.addEventListener('click', resetConversation);
    if (sidebarNewChatBtn) sidebarNewChatBtn.addEventListener('click', resetConversation);

    // Mobile sidebar toggle
    if (sidebarToggleBtn) sidebarToggleBtn.addEventListener('click', openSidebar);
    if (sidebarCloseBtn)  sidebarCloseBtn.addEventListener('click', closeSidebar);
    if (sidebarOverlay)   sidebarOverlay.addEventListener('click', closeSidebar);
  }

  // -------------------------------------------------------------------------
  // Sidebar Mobile
  // -------------------------------------------------------------------------
  function openSidebar() {
    state.sidebarOpen = true;
    sidebar.classList.add('open');
    sidebarOverlay.classList.add('visible');
    sidebarToggleBtn && sidebarToggleBtn.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden';
  }

  function closeSidebar() {
    state.sidebarOpen = false;
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('visible');
    sidebarToggleBtn && sidebarToggleBtn.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  }

  // -------------------------------------------------------------------------
  // Topic Selection
  // -------------------------------------------------------------------------
  function selectTopic(topicKey) {
    const topic = TOPICS[topicKey];
    if (!topic) return;

    // Close mobile sidebar after selection
    if (state.sidebarOpen) closeSidebar();

    // Update state
    const isNewTopic = state.selectedTopic !== topicKey;
    state.selectedTopic = topicKey;

    // Highlight active item in sidebar
    document.querySelectorAll('.topic-item').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.topic === topicKey);
    });

    // Update header breadcrumb
    headerTopicLabel.textContent = topic.label;

    // Update input badge
    inputTopicBadge.textContent = `📂 ${topic.label}`;
    inputTopicBadge.style.display = 'inline-flex';

    // If we haven't started a chat yet → show topic context card, hide welcome
    if (!state.hasStartedChat) {
      showTopicContext(topicKey, topic);
    }

    // Update chat textarea placeholder
    chatInput.placeholder = `Ask about ${topic.label}…`;
    chatInput.focus();
  }

  function showTopicContext(topicKey, topic) {
    // Hide welcome state
    if (welcomeState) welcomeState.style.display = 'none';

    // Populate context card
    tccIcon.innerHTML = topic.icon;
    tccTitle.textContent = topic.label;
    tccDesc.textContent = topic.description;

    // Build suggestion chips
    tccChips.innerHTML = '';
    topic.suggestions.forEach(q => {
      const chip = document.createElement('button');
      chip.className = 'suggestion-chip';
      chip.dataset.prompt = q;
      chip.type = 'button';
      chip.setAttribute('role', 'listitem');
      chip.textContent = q;
      tccChips.appendChild(chip);
    });

    topicContextCard.style.display = 'block';
  }

  // -------------------------------------------------------------------------
  // System Status Check
  // -------------------------------------------------------------------------
  async function checkSystemStatus() {
    try {
      const res = await fetch(KB_STATUS_ENDPOINT);
      if (!res.ok) throw new Error('Status check failed: ' + res.status);
      const data = await res.json();
      const chunks = data.total_chunks_indexed || 0;
      systemStatusChip.classList.add('ready');
      statusLabel.textContent = `Online · ${chunks} chunks indexed`;
      systemStatusChip.title = `Collection: ${data.collection_name} · ${chunks} chunks · ${data.embedding_model}`;
    } catch (err) {
      console.warn('Backend status check failed:', err);
      systemStatusChip.classList.add('error');
      statusLabel.textContent = 'Backend offline';
    }
  }

  // -------------------------------------------------------------------------
  // Input Handling
  // -------------------------------------------------------------------------
  function handleInputChange() {
    const len = chatInput.value.length;
    charCounter.textContent = `${len} / 2000`;
    charCounter.classList.toggle('warn', len > 1800);
    adjustTextareaHeight();
  }

  function adjustTextareaHeight() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
  }

  function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!state.isGenerating && chatInput.value.trim()) {
        chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    }
  }

  function handleFormSubmit(e) {
    e.preventDefault();
    if (state.isGenerating) return;
    const msg = chatInput.value.trim();
    if (!msg) return;
    chatInput.value = '';
    handleInputChange();
    sendMessage(msg);
  }

  // -------------------------------------------------------------------------
  // Send Message — Core Workflow
  // -------------------------------------------------------------------------
  async function sendMessage(userText) {
    state.isGenerating = true;
    updateControlsState();

    // Hide welcome & topic context on first message
    if (!state.hasStartedChat) {
      if (welcomeState) welcomeState.style.display = 'none';
      if (topicContextCard) topicContextCard.style.display = 'none';
      state.hasStartedChat = true;
    }

    renderUserMessage(userText);
    const loadingId = renderLoadingIndicator();
    scrollToBottom();

    try {
      // Build payload — include selected category for category-aware RAG
      const payload = {
        message: userText,
        conversation_history: state.conversationHistory
      };
      if (state.selectedTopic) {
        payload.category = state.selectedTopic;
      }

      const res = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      removeElementById(loadingId);
      renderAssistantMessage(data);

      // Update conversation history
      state.conversationHistory.push({ role: 'user',      content: userText    });
      state.conversationHistory.push({ role: 'assistant', content: data.answer });

      // Keep history bounded
      if (state.conversationHistory.length > 12) {
        state.conversationHistory = state.conversationHistory.slice(-12);
      }

    } catch (err) {
      console.error('Chat error:', err);
      removeElementById(loadingId);
      renderErrorMessage(err.message || 'Failed to connect to NovaTech Support API.');
    } finally {
      state.isGenerating = false;
      updateControlsState();
      scrollToBottom();
      chatInput.focus();
    }
  }

  // -------------------------------------------------------------------------
  // Render Functions
  // -------------------------------------------------------------------------

  function renderUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user-row';
    row.setAttribute('aria-label', 'Your message');
    row.innerHTML = `
      <div class="msg-content-wrapper">
        <div class="msg-bubble">${escapeHtml(text)}</div>
        <span class="msg-timestamp" aria-hidden="true">${formatTimestamp(new Date())}</span>
      </div>
      <div class="msg-avatar" aria-hidden="true" title="You">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
          <circle cx="12" cy="7" r="4"/>
        </svg>
      </div>
    `;
    chatFeed.appendChild(row);
  }

  function renderAssistantMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';
    row.setAttribute('aria-label', 'Support AI response');

    const formattedAnswer = formatMarkdownText(data.answer || '');

    let escalationHtml = '';
    if (data.needs_escalation) {
      escalationHtml = `
        <div class="escalation-banner" role="alert">
          <div class="escalation-header">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span>Escalated to Support Specialist</span>
          </div>
          <p class="escalation-desc">
            This inquiry has been flagged for human agent review.
            A NovaTech support specialist will assist you shortly.
          </p>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="msg-avatar" aria-hidden="true" title="NovaTech Support AI">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M2 17l10 5 10-5"            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M2 12l10 5 10-5"            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="msg-content-wrapper">
        <div class="msg-bubble" aria-live="polite">
          ${formattedAnswer}
          ${escalationHtml}
        </div>
        <span class="msg-timestamp" aria-hidden="true">${formatTimestamp(new Date())}</span>
      </div>
    `;

    chatFeed.appendChild(row);
  }

  function renderLoadingIndicator() {
    const id  = 'loadingIndicator_' + Date.now();
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';
    row.id        = id;
    row.setAttribute('aria-label', 'Support AI is typing');
    row.setAttribute('aria-live', 'polite');

    row.innerHTML = `
      <div class="msg-avatar" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M2 17l10 5 10-5"            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M2 12l10 5 10-5"            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="msg-content-wrapper">
        <div class="msg-bubble">
          <div class="typing-indicator" aria-label="Processing your question">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="loading-text">Searching knowledge base…</span>
          </div>
        </div>
      </div>
    `;

    chatFeed.appendChild(row);
    return id;
  }

  function renderErrorMessage(errText) {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';
    row.setAttribute('aria-label', 'Connection error');

    row.innerHTML = `
      <div class="msg-avatar" style="background:#450a0a;border-color:rgba(239,68,68,0.4);" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
      </div>
      <div class="msg-content-wrapper">
        <div class="error-bubble" role="alert">
          <strong>Connection Issue</strong>
          <p>${escapeHtml(errText)}</p>
          <p class="error-hint">Make sure the FastAPI server is running at <code>127.0.0.1:8000</code> and Ollama is active.</p>
        </div>
      </div>
    `;

    chatFeed.appendChild(row);
  }

  // -------------------------------------------------------------------------
  // Reset / New Chat
  // -------------------------------------------------------------------------
  function resetConversation() {
    if (state.isGenerating) return;

    state.conversationHistory = [];
    state.hasStartedChat = false;

    // Clear all chat messages
    chatFeed.innerHTML = '';

    // Restore welcome state
    if (welcomeState) {
      welcomeState.style.display = 'flex';
      chatFeed.appendChild(welcomeState);
    }

    // Hide topic context card
    if (topicContextCard) topicContextCard.style.display = 'none';

    // If a topic was selected, re-show context card for that topic
    if (state.selectedTopic && TOPICS[state.selectedTopic]) {
      if (welcomeState) welcomeState.style.display = 'none';
      showTopicContext(state.selectedTopic, TOPICS[state.selectedTopic]);
      chatFeed.appendChild(topicContextCard);
    }

    chatInput.value   = '';
    handleInputChange();
    chatInput.focus();
  }

  // -------------------------------------------------------------------------
  // UI Controls State
  // -------------------------------------------------------------------------
  function updateControlsState() {
    sendBtn.disabled    = state.isGenerating;
    chatInput.disabled  = state.isGenerating;
    if (clearChatBtn)     clearChatBtn.disabled    = state.isGenerating;
    if (sidebarNewChatBtn) sidebarNewChatBtn.disabled = state.isGenerating;
  }

  // -------------------------------------------------------------------------
  // Scroll
  // -------------------------------------------------------------------------
  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMain.scrollTop = chatMain.scrollHeight;
    });
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------
  function removeElementById(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function formatTimestamp(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g,  '&amp;')
      .replace(/</g,  '&lt;')
      .replace(/>/g,  '&gt;')
      .replace(/"/g,  '&quot;')
      .replace(/'/g,  '&#039;');
  }

  /**
   * Lightweight Markdown-to-HTML renderer
   * Handles: **bold**, `inline code`, - bullet lists, 1. ordered lists, paragraphs
   */
  function formatMarkdownText(rawText) {
    if (!rawText) return '';

    let text = escapeHtml(rawText);

    // Bold **text**
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Inline `code`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Line-by-line rendering
    const lines  = text.split('\n');
    let inList   = false;
    let listType = 'ul';
    const out    = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (line.startsWith('- ') || line.startsWith('* ')) {
        if (!inList || listType !== 'ul') {
          if (inList) out.push(`</${listType}>`);
          out.push('<ul>');
          inList   = true;
          listType = 'ul';
        }
        out.push(`<li>${line.substring(2)}</li>`);

      } else if (/^\d+\.\s/.test(line)) {
        if (!inList || listType !== 'ol') {
          if (inList) out.push(`</${listType}>`);
          out.push('<ol>');
          inList   = true;
          listType = 'ol';
        }
        out.push(`<li>${line.replace(/^\d+\.\s/, '')}</li>`);

      } else {
        if (inList) {
          out.push(`</${listType}>`);
          inList = false;
        }
        if (line.length > 0) out.push(`<p>${line}</p>`);
      }
    }

    if (inList) out.push(`</${listType}>`);

    return out.join('');
  }

  // -------------------------------------------------------------------------
  // Bootstrap
  // -------------------------------------------------------------------------
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
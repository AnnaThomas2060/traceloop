document.addEventListener("DOMContentLoaded", () => {
  const queryInput = document.getElementById("query-input");
  const submitBtn = document.getElementById("submit-btn");
  const statusBox = document.getElementById("status-box");
  const resultContainer = document.getElementById("result-container");
  const toolBadge = document.getElementById("tool-badge");
  const latencyBadge = document.getElementById("latency-badge");
  const chunkList = document.getElementById("chunk-list");
  const answerText = document.getElementById("answer-text");
  const traceLink = document.getElementById("trace-link");
  const thumbsUpBtn = document.getElementById("thumbs-up-btn");
  const thumbsDownBtn = document.getElementById("thumbs-down-btn");

  let currentTraceId = null;

  function setStatus(message, type) {
    if (!message) {
      statusBox.style.display = "none";
      statusBox.className = "status-message";
      statusBox.textContent = "";
      return;
    }
    statusBox.textContent = message;
    statusBox.className = `status-message ${type}`;
    statusBox.style.display = "block";
  }

  submitBtn.addEventListener("click", async () => {
    const query = queryInput.value.trim();
    if (!query) {
      setStatus("Query input cannot be empty. Please enter a technical question or message.", "error");
      return;
    }

    submitBtn.disabled = true;
    resultContainer.style.display = "none";
    setStatus("Executing query through agent and vector retriever. Please wait...", "info");

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Server returned an error status code.");
      }

      const data = await response.json();
      currentTraceId = data.trace_id;

      // Render tool badge
      toolBadge.textContent = data.tool_used;
      if (data.tool_used === "knowledge_retriever_tool") {
        toolBadge.className = "badge badge-retriever";
      } else {
        toolBadge.className = "badge badge-fallback";
      }

      // Render latency
      latencyBadge.textContent = `${data.latency_ms} ms`;

      // Render chunks chips
      chunkList.innerHTML = "";
      if (data.retrieved_chunk_ids && data.retrieved_chunk_ids.length > 0) {
        data.retrieved_chunk_ids.forEach((cid) => {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.textContent = cid;
          chunkList.appendChild(chip);
        });
      } else {
        const emptyChip = document.createElement("span");
        emptyChip.className = "chip";
        emptyChip.textContent = "None (direct tool)";
        chunkList.appendChild(emptyChip);
      }

      // Render answer
      answerText.textContent = data.answer;

      // Set trace link
      traceLink.href = data.trace_url;
      traceLink.style.display = "inline-block";

      // Reset feedback buttons
      thumbsUpBtn.className = "feedback-btn";
      thumbsDownBtn.className = "feedback-btn";

      setStatus("", "");
      resultContainer.style.display = "block";
    } catch (err) {
      setStatus(`Failed to process request: ${err.message}. Please verify service status.`, "error");
    } finally {
      submitBtn.disabled = false;
    }
  });

  thumbsUpBtn.addEventListener("click", () => {
    thumbsUpBtn.className = "feedback-btn active";
    thumbsDownBtn.className = "feedback-btn";
    setStatus("Positive feedback recorded for this trace session.", "info");
  });

  thumbsDownBtn.addEventListener("click", () => {
    thumbsDownBtn.className = "feedback-btn active";
    thumbsUpBtn.className = "feedback-btn";
    setStatus("Negative feedback recorded. This trace is eligible for regression dataset synchronization.", "info");
  });
});

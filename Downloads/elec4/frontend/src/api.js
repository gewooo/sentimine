const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error("Backend is not running. Start the API server on http://localhost:8000.");
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail || payload.error || payload.message : payload;
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return payload;
}

export function classifySentence(sentence, algorithm = null) {
  // classifySingleSentenceRequest: Classify one typed sentence through the API or offline fallback.
  return request("/api/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence, algorithm }),
  }).catch(() => fallbackClassify(sentence));
}

export function classifyDataset(file, algorithm = null) {
  // classifyUploadedDatasetRequest: Send uploaded dataset file to the API or summarize it locally.
  const formData = new FormData();
  formData.append("file", file);
  
  const path = algorithm ? `/api/classify-dataset?algorithm=${encodeURIComponent(algorithm)}` : "/api/classify-dataset";

  return request(path, {
    method: "POST",
    body: formData,
  }).catch(() => summarizeUploadedFile(file));
}

export function getModelComparison() {
  // loadModelComparisonMetrics: Load model scores for the comparison graph.
  return request("/api/models/comparison").catch(() => ({
    best_model: "SVM",
    models: [
      { name: "Logistic Regression", accuracy: 0.8014, precision: 0.8069, recall: 0.8014, f1: 0.801 },
      { name: "Naive Bayes", accuracy: 0.7769, precision: 0.794, recall: 0.7769, f1: 0.771 },
      { name: "SVM", accuracy: 0.8259, precision: 0.828, recall: 0.8259, f1: 0.825 },
      { name: "Random Forest", accuracy: 0.7864, precision: 0.79, recall: 0.7864, f1: 0.783 },
    ],
  }));
}

export function getOriginalDataset(algorithm = null) {
  // loadOriginalFilteredDataset: Load datasets/unlabeled_dataset.csv as the default dashboard source.
  const params = new URLSearchParams();
  if (algorithm) params.append("algorithm", algorithm);
  params.append("_t", Date.now()); // Cache buster
  const path = `/api/dataset/original?${params.toString()}`;
  return request(path).catch(loadLocalOriginalDataset);
}

export function downloadDatasetTemplate() {
  // downloadDatasetTemplateCsv: Download fixed CSV headers for user-uploaded datasets.
  const csv = [
    "Category,Product Name,Review",
    "Food,Sample Ice Cream,this ice cream is yummy and delicious",
    "Gadgets,Sample Keyboard,the item arrived damaged and I am disappointed",
    "Home,Sample Lamp,the delivery was late and the seller was rude",
  ].join("\n");
  downloadCsv("sentimine_dataset_template.csv", csv);
}

function fallbackClassify(sentence) {
  // offlineSentenceClassifier: Keep sentence classification working when the backend is unavailable.
  const happyWords = new Set(["good", "great", "excellent", "amazing", "love", "loved", "like", "delicious", "perfect", "happy", "fast", "safe", "thank", "thanks", "nice", "best", "quality", "trusted", "recommend", "yummy", "tasty", "fresh", "awesome", "wonderful", "pleased"]);
  const sadWords = new Set(["sad", "disappointed", "disappointing", "poor", "broken", "late", "missing", "wrong", "damaged", "bad", "waste", "slow", "issue", "problem", "terrible", "unhappy", "return"]);
  const angryWords = new Set(["angry", "mad", "hate", "hated", "rude", "scam", "fake", "awful", "worst", "annoying", "frustrated", "unhelpful", "refund", "complaint", "useless"]);
  const words = sentence.match(/[a-zA-Z][a-zA-Z']*/g) || [];
  const counts = words.reduce((acc, word) => {
    const clean = word.toLowerCase();
    if (happyWords.has(clean)) acc.happy += 1;
    if (sadWords.has(clean)) acc.sad += 1;
    if (angryWords.has(clean)) acc.angry += 1;
    return acc;
  }, { happy: 0, sad: 0, angry: 0 });
  const emotion = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][1] > 0
    ? Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
    : "happy";
  const probabilities = {
    happy: emotion === "happy" ? 0.84 : 0.08,
    sad: emotion === "sad" ? 0.84 : 0.08,
    angry: emotion === "angry" ? 0.84 : 0.08,
  };

  return {
    emotion,
    probabilities,
    words: words.map((word) => {
      const clean = word.toLowerCase();
      const scores = { happy: 0.04, sad: 0.04, angry: 0.04 };
      if (happyWords.has(clean)) scores.happy = 0.95;
      if (sadWords.has(clean)) scores.sad = 0.95;
      if (angryWords.has(clean)) scores.angry = 0.95;
      return { word, scores };
    }),
  };
}

async function loadLocalOriginalDataset() {
  // offlineOriginalDatasetLoader: Read the bundled filtered dataset for offline presentation mode.
  const response = await fetch("/filtered_dataset.csv");
  if (!response.ok) throw new Error("Original dataset file is missing.");
  const text = await response.text();
  const rows = parseCsv(text);
  return summarizeRows(rows, true);
}

async function summarizeUploadedFile(file) {
  // offlineUploadedDatasetSummary: Build dashboard data from an uploaded CSV without the backend.
  if (!file.name.toLowerCase().endsWith(".csv")) {
    throw new Error("Offline mode supports CSV uploads. Use the downloadable CSV template.");
  }
  const text = await file.text();
  return summarizeRows(parseCsv(text), false);
}

function parseCsv(text) {
  // parseCsvRows: Convert CSV text into row objects while respecting quoted commas.
  const rows = [];
  let current = "";
  let row = [];
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(current.trim());
      current = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(current.trim());
      if (row.some(Boolean)) rows.push(row);
      row = [];
      current = "";
    } else {
      current += char;
    }
  }

  if (current || row.length) {
    row.push(current.trim());
    if (row.some(Boolean)) rows.push(row);
  }

  const headers = rows.shift()?.map((header) => header.replace(/^"|"$/g, "")) || [];
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] || ""])));
}

function summarizeRows(rows, hasEmotion) {
  // summarizeDatasetRows: Build KPI, chart, ranking, and word-cloud data from review rows.
  const normalizedRows = rows.map((row) => {
    const review = row.Review || row["Customer Review"] || row["Customer Review (English)"] || row.review || "";
    const predicted = hasEmotion && row.Emotion
      ? { emotion: normalizeEmotionName(row.Emotion) }
      : fallbackClassify(review);
    return {
      category: row.Category || row.category || "Uncategorized",
      product: row["Product Name"] || row.product || row.Product || "Unknown product",
      review,
      emotion: normalizeEmotionName(predicted.emotion),
    };
  }).filter((row) => row.review || row.emotion);

  const summary = normalizedRows.reduce((acc, row) => {
    acc.total += 1;
    acc[`${row.emotion}_count`] += 1;
    return acc;
  }, { total: 0, happy_count: 0, sad_count: 0, angry_count: 0 });

  const groupBy = (field) => Object.values(normalizedRows.reduce((acc, row) => {
    const key = row[field] || "Unknown";
    acc[key] ||= { name: key, happy: 0, sad: 0, angry: 0 };
    acc[key][row.emotion] += 1;
    return acc;
  }, {})).sort((a, b) => (b.happy + b.sad + b.angry) - (a.happy + a.sad + a.angry)).slice(0, 10);

  const counts = {};
  normalizedRows.forEach((row) => {
    tokenize(row.review).forEach((word) => {
      const key = `${row.emotion}:${word}`;
      counts[key] ||= { word, emotion: row.emotion, count: 0 };
      counts[key].count += 1;
    });
  });

  return {
    results: normalizedRows,
    summary,
    top_products: groupBy("product"),
    top_categories: groupBy("category"),
    wordcloud: Object.values(counts).sort((a, b) => b.count - a.count).slice(0, 90),
  };
}

function normalizeEmotionName(value) {
  const emotion = String(value).toLowerCase();
  if (emotion.includes("sad")) return "sad";
  if (emotion.includes("ang")) return "angry";
  return "happy";
}

function tokenize(text) {
  const stop = new Set(["the", "and", "for", "with", "this", "that", "was", "were", "are", "you", "but", "not", "very"]);
  return (text.match(/[a-zA-Z][a-zA-Z']+/g) || [])
    .map((word) => word.toLowerCase())
    .filter((word) => word.length > 2 && !stop.has(word));
}

function downloadCsv(filename, content) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

// ============================================
// IMPORTS
// ============================================
import React, { useEffect, useMemo, useState } from "react";
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";
import cloud from "d3-cloud";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowRight,
  Brain,
  CheckCircle2,
  FileSpreadsheet,
  Layers3,
  Search,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { classifyDataset, classifySentence, downloadDatasetTemplate, getModelComparison, getOriginalDataset } from "./api";
import Header from "./components/Header";
import {
  EMOTION_COLORS,
  EMOTION_EMOJIS,
  EMOTION_KEYS,
  EMOTION_LABELS,
  METRIC_COLORS,
  normalizeEmotion,
} from "./constants";
import "./App.css";

// ============================================
// CONFIGURATION & CONSTANTS
// ============================================
const tabs = {
  home: "Home",
  process: "Classification Process",
  training: "Model Training",
};

const DEFAULT_MODELS = ["SVM", "Logistic Regression", "Random Forest", "Naive Bayes"];

// Helper function to generate placeholder items for empty state
const makeZeroTopItems = (group) => Array.from({ length: 10 }, (_, index) => ({
  name: `${group === "product" ? "Product" : "Category"} ${index + 1}`,
  happy: 0,
  sad: 0,
  angry: 0,
}));

// Ghost data for word cloud empty state
const GHOST_WORDS = [
  { word: "service", count: 8, emotion: "happy" },
  { word: "quality", count: 7, emotion: "sad" },
  { word: "delivery", count: 6, emotion: "angry" },
  { word: "support", count: 5, emotion: "happy" },
  { word: "price", count: 4, emotion: "sad" },
];

// Example sentences for quick testing
const SAMPLE_SENTENCES = [
  "this food is yummy",
  "this fish tastes bad",
  "the delivery was fast and safe",
];

// ============================================
// SHARED COMPONENTS
// ============================================

// Reusable tooltip component for charts
function ChartTooltip({ active, payload, label, formatter }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">{label || payload[0]?.name}</p>
      {payload.map((item) => (
        <p key={item.dataKey || item.name} style={{ color: item.color || item.fill }}>
          {item.name}: <strong>{formatter ? formatter(item.value, item) : item.value}</strong>
        </p>
      ))}
    </div>
  );
}

// ============================================
// HOME PAGE & RELATED COMPONENTS
// ============================================

// Main home page: single sentence classification, dataset dashboard, and model comparison
function HomePage() {
  const [sentence, setSentence] = useState("");
  const [singleResult, setSingleResult] = useState(null);
  const [singleStatus, setSingleStatus] = useState("idle");
  const [singleError, setSingleError] = useState("");
  const [showOriginalDataset, setShowOriginalDataset] = useState(true);
  const [datasetResult, setDatasetResult] = useState(null);
  const [originalDataset, setOriginalDataset] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [datasetStatus, setDatasetStatus] = useState("idle");
  const [datasetError, setDatasetError] = useState("");
  const [modelComparison, setModelComparison] = useState(null);
  const [modelStatus, setModelStatus] = useState("loading");
  const [modelError, setModelError] = useState("");
  const [selectedModel, setSelectedModel] = useState("SVM");
  const [uploadedFile, setUploadedFile] = useState(null); // Store actual file for re-classification on model toggle

  // fetchHomePageInitialData: Load model comparison metrics.
  useEffect(() => {
    let ignore = false;

    getModelComparison()
      .then((data) => {
        if (!ignore) {
          setModelComparison(data);
          setModelStatus("success");
        }
      })
      .catch((error) => {
        if (!ignore) {
          setModelError(error.message);
          setModelStatus("error");
        }
      });

    return () => {
      ignore = true;
    };
  }, []);

  // Sync dashboard data when selected model or dataset source changes
  useEffect(() => {
    let ignore = false;

    if (uploadedFile) {
      // Re-run classification for the uploaded file with the new model
      setDatasetStatus("loading");
      classifyDataset(uploadedFile, selectedModel)
        .then((data) => {
          if (!ignore) {
            setDatasetResult(data);
            setDatasetStatus("success");
          }
        })
        .catch((error) => {
          if (!ignore) {
            setDatasetError(error.message);
            setDatasetStatus("error");
          }
        });
    } else if (showOriginalDataset) {
      // Re-fetch predictions for the original dataset using the new model
      setDatasetStatus("loading");
      getOriginalDataset(selectedModel)
        .then((data) => {
          if (!ignore) {
            // Store the initial original dataset only once if needed, 
            // but for visualizations we always use the results of the selected model.
            setOriginalDataset(data);
            setDatasetResult(data);
            setDatasetStatus("success");
          }
        })
        .catch((error) => {
          if (!ignore) {
            setDatasetError(error.message);
            setDatasetStatus("error");
          }
        });
    }

    return () => {
      ignore = true;
    };
  }, [selectedModel, showOriginalDataset, uploadedFile]);

  const handleClassify = async (event) => {
    event.preventDefault();
    if (!sentence.trim()) return;

    setSingleStatus("loading");
    setSingleError("");
    try {
      const response = await classifySentence(sentence.trim(), selectedModel);
      setSingleResult(response);
      setSingleStatus("success");
    } catch (error) {
      setSingleError(error.message);
      setSingleStatus("error");
    }
  };

  const resetSentence = () => {
    setSentence("");
    setSingleResult(null);
    setSingleStatus("idle");
    setSingleError("");
  };

  // reloadChartsFromUploadedDataset: Reload charts based on the uploaded dataset.
  const handleDatasetFile = (file) => {
    if (!file) return;
    setUploadedFile(file);
    setUploadedFileName(file.name);
    setShowOriginalDataset(false);
  };

  const handleDatasetUpload = async (event) => {
    await handleDatasetFile(event.target.files?.[0]);
    event.target.value = "";
  };

  return (
    <div className="page-stack">
      <section className="hero-section">
        <form className="sentence-form" onSubmit={handleClassify}>
          <input
            value={sentence}
            onChange={(event) => setSentence(event.target.value)}
            placeholder="Enter a single sentence"
            aria-label="Enter a single sentence"
          />
          <button className="btn btn-primary" disabled={singleStatus === "loading" || !sentence.trim()}>
            {singleStatus === "loading" && <span className="spinner spinner-light" />}
            {singleStatus === "loading" ? "Classifying" : "Classify"}
          </button>
        </form>

        <div className="sample-sentences" aria-label="Sample sentences">
          {SAMPLE_SENTENCES.map((sample) => (
            <button
              type="button"
              key={sample}
              onClick={() => {
                setSentence(sample);
                setSingleResult(null);
                setSingleError("");
                setSingleStatus("idle");
              }}
            >
              {sample}
            </button>
          ))}
        </div>

        {singleError && <p className="inline-error">{singleError}</p>}

        <SingleSentenceResult result={singleResult} status={singleStatus} onReset={resetSentence} />
      </section>

      <DatasetSection
        result={datasetResult}
        status={datasetStatus}
        error={datasetError}
        showOriginalDataset={showOriginalDataset}
        onDismissOriginal={() => setShowOriginalDataset(false)}
        onRestoreOriginal={() => {
          setShowOriginalDataset(true);
          setUploadedFileName("");
          setUploadedFile(null);
        }}
        uploadedFileName={uploadedFileName}
        onClearUploaded={() => {
          setUploadedFileName("");
          setUploadedFile(null);
          setShowOriginalDataset(false);
          setDatasetResult(null);
          setDatasetStatus("idle");
          setDatasetError("");
        }}
        onDropUpload={handleDatasetFile}
        onUpload={handleDatasetUpload}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />

      <ModelComparisonSection data={modelComparison} status={modelStatus} error={modelError} />
    </div>
  );
}

// Displays emotion result card with confidence percentages for a single classified sentence
function SingleSentenceResult({ result, status, onReset }) {
  const emotion = normalizeEmotion(result?.emotion);
  const confidence = Math.round(Number(result?.probabilities?.[emotion] || 0) * 100);
  const probabilityRows = EMOTION_KEYS.map((key) => ({
    key,
    label: capitalize(EMOTION_LABELS[key]),
    value: Math.round(Number(result?.probabilities?.[key] || 0) * 100),
  }));

  if (!result) {
    return (
      <div className="emotion-summary-card zero-state-summary">
        <div className="emotion-orb-block">
          <span className="emotion-orb" />
          <strong>Your result</strong>
          <small>will appear here</small>
        </div>
        <div className="confidence-table muted-confidence">
          {status === "loading" ? (
            <span className="inline-loading"><span className="spinner" /> Reading emotion signals</span>
          ) : (
            <>
              <div className="confidence-head"><span>Emotion</span><span>Percentage</span></div>
              {EMOTION_KEYS.map((key) => (
                <div className="confidence-row" key={key}>
                  <span><span className="emotion-mark">{EMOTION_EMOJIS[key]}</span>{capitalize(EMOTION_LABELS[key])}</span>
                  <span>-</span>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="result-wrap">
      <div className="emotion-summary-card">
        <div className="emotion-orb-block">
          <span className="result-emoji" aria-hidden="true">{EMOTION_EMOJIS[emotion]}</span>
          <strong>{capitalize(EMOTION_LABELS[emotion])}</strong>
          <small>{confidence}% sure</small>
        </div>
        <div className="confidence-table">
          <div className="confidence-head"><span>Emotion</span><span>Percentage</span></div>
          {probabilityRows.map((row) => (
            <div className={`confidence-row ${row.key === emotion ? "active" : ""}`} key={row.key}>
              <span><span className="emotion-mark">{EMOTION_EMOJIS[row.key]}</span>{row.label}</span>
              <span>{row.value}%</span>
            </div>
          ))}
        </div>
      </div>
      <button className="btn btn-ghost try-again" onClick={onReset}>
        Try Another
      </button>
    </div>
  );
}

// Dataset upload section with file handling and dashboard rendering
function DatasetSection({ result, status, error, showOriginalDataset, uploadedFileName, onDismissOriginal, onRestoreOriginal, onClearUploaded, onDropUpload, onUpload, selectedModel, onModelChange }) {
  // exportDatasetPdfReport: Export the dataset dashboard and visualizations as a PDF-ready report.
  const exportDatasetSection = async () => {
    const dashboardElement = document.querySelector(".dashboard-grid");
    if (!dashboardElement || !result) return;

    try {
      const pdf = new jsPDF("p", "mm", "a4");
      const pageWidth = pdf.internal.pageSize.getWidth();
      
      // 1. Add Header & Info
      pdf.setFontSize(22);
      pdf.setTextColor(14, 165, 233);
      pdf.text("Sentimine Sentiment Report", 20, 25);
      
      pdf.setFontSize(12);
      pdf.setTextColor(100, 100, 100);
      pdf.text(`Generated on: ${new Date().toLocaleString()}`, 20, 32);
      pdf.text(`Model Used: ${selectedModel}`, 20, 38);

      pdf.setDrawColor(240, 240, 240);
      pdf.line(20, 45, pageWidth - 20, 45);

      // 2. Add Dataset Info
      pdf.setFontSize(16);
      pdf.setTextColor(50, 50, 50);
      pdf.text("Dataset Overview", 20, 58);
      
      pdf.setFontSize(11);
      const rowCount = result?.summary?.total || 0;
      const fileName = uploadedFileName || result?.dataset_name || "unlabeled_dataset.csv";
      
      pdf.text(`File Name: ${fileName}`, 25, 68);
      pdf.text(`Total Data Rows: ${rowCount.toLocaleString()} (excluding header row)`, 25, 74);
      pdf.text(`Status: Successfully Classified`, 25, 80);

      // 3. Capture Dashboard Charts
      const canvas = await html2canvas(dashboardElement, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#ffffff",
        logging: false,
        onclone: (clonedDoc) => {
          // Fix for "Unsupported color function color" error in html2canvas
          // This aggressively finds and removes modern CSS color functions from style tags
          const styleTags = clonedDoc.getElementsByTagName("style");
          for (let i = 0; i < styleTags.length; i++) {
            let css = styleTags[i].innerHTML;
            if (css.includes("color(") || css.includes("oklch(") || css.includes("oklab(")) {
              // Replace modern color functions with standard hex fallback
              styleTags[i].innerHTML = css.replace(/color\([^)]+\)/g, "#0ea5e9")
                                         .replace(/oklch\([^)]+\)/g, "#0ea5e9")
                                         .replace(/oklab\([^)]+\)/g, "#0ea5e9");
            }
          }
          // Also check inline styles
          const elements = clonedDoc.getElementsByTagName("*");
          for (let i = 0; i < elements.length; i++) {
            const el = elements[i];
            if (el.style) {
              if (el.style.color && (el.style.color.includes("color(") || el.style.color.includes("oklch("))) el.style.color = "#000000";
              if (el.style.backgroundColor && (el.style.backgroundColor.includes("color(") || el.style.backgroundColor.includes("oklch("))) el.style.backgroundColor = "transparent";
            }
          }
        }
      });
      
      const imgData = canvas.toDataURL("image/png");
      const imgWidth = pageWidth - 40;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      
      // Check if image fits on first page, otherwise add new page
      if (imgHeight > 180) {
        pdf.addPage();
        pdf.addImage(imgData, "PNG", 20, 20, imgWidth, imgHeight);
      } else {
        pdf.addImage(imgData, "PNG", 20, 90, imgWidth, imgHeight);
      }

      pdf.save(`Sentimine_Report_${fileName.split(".")[0]}.pdf`);
    } catch (err) {
      console.error("PDF Export failed:", err);
      // Fallback for color function error or other canvas failures
      if (err.message?.includes("color") || err.message?.includes("parse")) {
        alert("Encountered a styling incompatibility. Attempting a simplified export...");
        // Secondary attempt with simplified styles
        try {
          const pdf = new jsPDF("p", "mm", "a4");
          pdf.text("Sentimine Report (Simplified View)", 20, 20);
          pdf.text(`File: ${uploadedFileName || result?.dataset_name || "dataset.csv"}`, 20, 30);
          pdf.text(`Total Rows: ${result?.summary?.total || 0}`, 20, 40);
          pdf.save("Sentimine_Report_Basic.pdf");
        } catch (innerErr) {
          alert(`Failed to generate PDF: ${innerErr.message}`);
        }
      } else {
        alert(`Failed to generate PDF: ${err.message}. Please try again.`);
      }
    }
  };

  return (
    <section className="section dataset-section">
      <div className="section-heading section-heading-row">
        <div>
          <p className="eyebrow">Classifying a dataset?</p>
          <div className="model-selection-header">
            <h2 className="single-line-heading">Choose your preferred model</h2>
            <div className="model-toggle-wrap">
              <Toggle
                value={selectedModel}
                options={[
                  { value: "SVM", label: "SVM 👑" },
                  { value: "Logistic Regression", label: "Logistic Regression" },
                  { value: "Naive Bayes", label: "Naive Bayes" },
                  { value: "Random Forest", label: "Random Forest" },
                ]}
                onChange={onModelChange}
              />
            </div>
          </div>
        </div>
        <button className="btn btn-ghost export-btn" type="button" onClick={exportDatasetSection}>
          Export PDF
        </button>
      </div>

      {showOriginalDataset || uploadedFileName ? (
        <div className="dataset-source">
          <div>
            <FileSpreadsheet size={22} />
            <div>
              <strong>{uploadedFileName ? "Uploaded dataset" : "Original dataset"}</strong>
              <p>{uploadedFileName ? `${uploadedFileName} is now powering the dashboard visualizations.` : `Using datasets/${result?.dataset_name || "unlabeled_dataset.csv"} for the default dashboard source.`}</p>
            </div>
          </div>
          <button className="icon-btn" onClick={uploadedFileName ? onClearUploaded : onDismissOriginal} aria-label={uploadedFileName ? "Clear uploaded dataset" : "Dismiss original dataset"}>
            <X size={18} />
          </button>
        </div>
      ) : (
        <div className="upload-row">
          <button className="btn btn-ghost template-btn" type="button" onClick={downloadDatasetTemplate}>
            Download template
          </button>
          <label
            className="upload-box"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              onDropUpload(event.dataTransfer.files?.[0]);
            }}
          >
            <Upload size={24} />
            <span>{status === "loading" ? "Classifying dataset..." : "Upload CSV or Excel"}</span>
            <small>Drag and drop a file here, or click to browse</small>
            <input type="file" accept=".csv,.xlsx,.xls" onChange={onUpload} disabled={status === "loading"} />
          </label>
          <button className="btn btn-ghost restore-original-btn" type="button" onClick={onRestoreOriginal}>
            Reset original
          </button>
        </div>
      )}

      {error && <p className="inline-error">{error}</p>}
      <DatasetDashboard data={result} status={status} />
    </section>
  );
}

// Dashboard with KPIs, emotion charts, top products/categories, and word cloud
function DatasetDashboard({ data, status }) {
  const [limit, setLimit] = useState(5);
  const [group, setGroup] = useState("product");
  const [productEmotion, setProductEmotion] = useState("all");
  const [wordEmotion, setWordEmotion] = useState("all");
  const hasData = Boolean(data);
  const summary = data?.summary || {};
  // emotionCountBarGraphData: Prepare happy, sad, and angry counts for the bar graph.
  const emotionData = EMOTION_KEYS.map((key) => ({
    name: EMOTION_LABELS[key],
    key,
    value: Number(summary[`${key}_count`] || 0),
  }));
  const total = Number(summary.total || emotionData.reduce((sum, item) => sum + item.value, 0));
  // topProductsCategoriesByEmotionData: Rank products or categories by all emotions or one selected emotion.
  const rankedItems = hasData
    ? normalizeTopItems(group === "product" ? data.top_products : data.top_categories)
        .sort((a, b) => {
          if (productEmotion === "all") return (b.happy + b.sad + b.angry) - (a.happy + a.sad + a.angry);
          return b[productEmotion] - a[productEmotion];
        })
        .slice(0, limit)
    : makeZeroTopItems(group).slice(0, limit);
  const filteredRankedItems = rankedItems.map((item) => {
    if (productEmotion === "all") return item;
    return { ...item, happy: productEmotion === "happy" ? item.happy : 0, sad: productEmotion === "sad" ? item.sad : 0, angry: productEmotion === "angry" ? item.angry : 0 };
  });
  const words = hasData
    ? normalizeWords(data.wordcloud).filter((item) => wordEmotion === "all" || item.emotion === wordEmotion)
    : GHOST_WORDS;
  // emotionShareDonutChartData: Keep the donut chart visible even before real upload data exists.
  const pieData = total > 0 ? emotionData : emotionData.map((item) => ({ ...item, value: 1 }));
  const isLoading = status === "loading";

  return (
    <div className="dashboard-grid">
      {isLoading && <LoadingPanel label="Classifying dataset and preparing charts" />}

      <div className="kpi-grid">
        <Kpi label="Total Reviews" value={total} />
        {emotionData.map((item) => (
          <Kpi key={item.key} label={`Total of ${capitalize(EMOTION_LABELS[item.key])}`} value={item.value} emotion={item.key} />
        ))}
      </div>

      <div className="chart-grid two">
        <ChartCard title="Emotion Counts">
          <div className="chart-shell">
            {!hasData && <EmptyChartMessage>Upload a dataset to see results</EmptyChartMessage>}
            <ResponsiveContainer width="100%" height={260}>
            {/* emotionCountBarGraph: Display emotion counts using dashboard data. */}
            <BarChart data={emotionData}>
              <CartesianGrid vertical={false} stroke="rgba(148, 163, 184, 0.14)" />
              <XAxis dataKey="name" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip content={<ChartTooltip formatter={(value) => Number(value).toLocaleString()} />} />
              <Bar dataKey="value" name="Count" radius={[8, 8, 0, 0]}>
                {emotionData.map((item) => (
                  <Cell key={item.key} fill={EMOTION_COLORS[item.key]} />
                ))}
              </Bar>
            </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Emotion Share">
          <div className="chart-shell">
            {!hasData && <EmptyChartMessage>Upload a dataset to see results</EmptyChartMessage>}
            <ResponsiveContainer width="100%" height={260}>
            {/* emotionShareDonutChart: Display percentage share for each emotion. */}
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={62} outerRadius={96} paddingAngle={3}>
                {pieData.map((item) => (
                  <Cell key={item.key} fill={EMOTION_COLORS[item.key]} fillOpacity={total > 0 ? 0.9 : 0.18} />
                ))}
              </Pie>
              <Tooltip content={<ChartTooltip formatter={(value) => `${((Number(value) / total) * 100 || 0).toFixed(1)}%`} />} />
            </PieChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      <ChartCard
        title="Top Products/Categories by Emotion"
        actions={
          <>
            <Toggle
              value={productEmotion}
              options={[
                { value: "all", label: "All" },
                { value: "happy", label: "Happy" },
                { value: "sad", label: "Sad" },
                { value: "angry", label: "Angry" },
              ]}
              onChange={setProductEmotion}
            />
            <Toggle value={limit} options={[5, 10]} onChange={setLimit} />
            <Toggle
              value={group}
              options={[
                { value: "product", label: "Product" },
                { value: "category", label: "Category" },
              ]}
              onChange={setGroup}
            />
          </>
        }
      >
        <div className="chart-shell">
          {!hasData && <EmptyChartMessage>Upload a dataset to see results</EmptyChartMessage>}
          <ResponsiveContainer width="100%" height={340}>
          {/* topProductsCategoriesByEmotionChart: Display top products or categories by selected emotion. */}
          <BarChart data={filteredRankedItems}>
            <CartesianGrid vertical={false} stroke="rgba(148, 163, 184, 0.14)" />
            <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} angle={-12} textAnchor="end" height={80} />
            <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
            <Tooltip content={<ChartTooltip formatter={(value) => Number(value).toLocaleString()} />} />
            {EMOTION_KEYS.map((key) => (
              <Bar key={key} dataKey={key} name={EMOTION_LABELS[key]} fill={EMOTION_COLORS[key]} radius={[6, 6, 0, 0]} />
            ))}
          </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <ChartCard title="Category vs Emotion Heatmap">
        <div className="chart-shell">
          {!hasData && <EmptyChartMessage>Upload a dataset to see results</EmptyChartMessage>}
          <CategoryEmotionHeatmap data={data?.category_emotion_matrix} />
        </div>
      </ChartCard>

      <ChartCard
        title="Word Cloud"
        actions={
          <Toggle
            value={wordEmotion}
            options={[
              { value: "all", label: "All" },
              { value: "happy", label: "Happy" },
              { value: "sad", label: "Sad" },
              { value: "angry", label: "Angry" },
            ]}
            onChange={setWordEmotion}
          />
        }
      >
        {/* emotionWordCloud: Display compact emotion-colored terms from classified reviews. */}
        <WordCloud words={words} isEmpty={!hasData} />
      </ChartCard>
    </div>
  );
}

// ============================================
// DASHBOARD UTILITY COMPONENTS
// ============================================

// Key performance indicator card for dashboard
function Kpi({ label, value, emotion }) {
  return (
    <div className="kpi-card" style={emotion ? { "--kpi": EMOTION_COLORS[emotion] } : undefined}>
      <span>{emotion ? EMOTION_EMOJIS[emotion] : <Layers3 size={20} />}</span>
      <div>
        <strong>{Number(value || 0).toLocaleString()}</strong>
        <p>{label}</p>
      </div>
    </div>
  );
}

// Wrapper for chart content with title and optional action buttons
function ChartCard({ title, actions, children }) {
  return (
    <div className="chart-card">
      <div className="chart-card-head">
        <h3>{title}</h3>
        {actions && <div className="chart-actions">{actions}</div>}
      </div>
      {children}
    </div>
  );
}

// Loading state overlay with spinner and label
function LoadingPanel({ label }) {
  return (
    <div className="loading-panel">
      <span className="spinner" />
      <p>{label}</p>
    </div>
  );
}

// Centered message shown when chart has no data
function EmptyChartMessage({ children }) {
  return <div className="empty-chart-message">{children}</div>;
}

// Button group for filtering and toggling chart options
function Toggle({ value, options, onChange }) {
  return (
    <div className="toggle">
      {options.map((option) => {
        const item = typeof option === "object" ? option : { value: option, label: `Top ${option}` };
        return (
          <button type="button" key={item.value} className={value === item.value ? "active" : ""} onClick={() => onChange(item.value)}>
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

// Generates d3-cloud layout for emotion-colored word cloud visualization
function WordCloud({ words, isEmpty }) {
  // compactEmotionWordCloudLayout: Use d3-cloud to place words in a stable compact layout.
  const arrangedWords = useMemo(() => [...words].sort((a, b) => b.count - a.count), [words]);
  const max = Math.max(...arrangedWords.map((item) => item.count), 1);
  const [layoutWords, setLayoutWords] = useState([]);

  useEffect(() => {
    let cancelled = false;
    const random = seededRandom(2026);
    const width = 940;
    const height = 290;

    const layout = cloud()
      .size([width, height])
      .words(arrangedWords.slice(0, 70).map((item) => ({
        ...item,
        text: item.word,
        size: 13 + (item.count / max) * 44,
      })))
      .padding(5)
      .rotate(0)
      .random(random)
      .font("Manrope")
      .fontWeight(800)
      .fontSize((item) => item.size)
      .on("end", (computedWords) => {
        if (!cancelled) setLayoutWords(computedWords);
      })
      .start();

    return () => {
      cancelled = true;
      layout.stop();
    };
  }, [arrangedWords, max]);

  if (!words.length) return <p className="muted-center">No words returned yet.</p>;

  return (
    <div className={`word-cloud word-cloud-library ${isEmpty ? "word-cloud-empty" : ""}`}>
      {isEmpty && <EmptyChartMessage>Upload a dataset to build the word cloud</EmptyChartMessage>}
      <svg className="word-cloud-svg" viewBox="-470 -145 940 290" role="img" aria-label="Emotion word cloud">
        {layoutWords.map((item) => (
          <text
            key={`${item.word}-${item.emotion}`}
            className="cloud-word-svg"
            textAnchor="middle"
            transform={`translate(${item.x}, ${item.y})`}
            style={{
              fill: EMOTION_COLORS[item.emotion] || "var(--text-muted)",
              fontSize: `${item.size}px`,
              fontWeight: 800,
            }}
          >
            <title>{`${item.word}: ${item.count.toLocaleString()} mentions`}</title>
            {item.word}
          </text>
        ))}
      </svg>
    </div>
  );
}

// ============================================
// MODEL COMPARISON SECTION
// ============================================

// Displays grouped bar chart comparing model accuracy, precision, recall, and F1-score
function ModelComparisonSection({ data, status, error }) {
  const models = data?.models || [];
  const computedBest = models.length ? [...models].sort((a, b) => Number(b.f1 || 0) - Number(a.f1 || 0))[0]?.name : "";
  const bestModel = data?.best_model || computedBest;
  const sourceModels = models.length ? models : DEFAULT_MODELS.map((name) => ({ name, accuracy: 0, precision: 0, recall: 0, f1: 0 }));
  // modelComparisonGroupedBarChartData: Convert API metrics into percentages for the grouped bar chart.
  const chartData = sourceModels.map((model) => ({
    name: model.name,
    accuracy: Number(model.accuracy || 0) * 100,
    precision: Number(model.precision || 0) * 100,
    recall: Number(model.recall || 0) * 100,
    f1: Number(model.f1 || 0) * 100,
  }));

  return (
    <section className="section">
      <div className="section-heading">
        <p className="eyebrow">Model comparison</p>
        <h2>Best model ({bestModel || "SVM"}) vs Others</h2>
      </div>

      {error && <p className="inline-error">{error}</p>}

      <div className="chart-card">
        {status === "loading" && <LoadingPanel label="Loading model comparison metrics" />}
        <div className="chart-shell">
          {!models.length && <EmptyChartMessage>Model metrics will load from the backend</EmptyChartMessage>}
          <ResponsiveContainer width="100%" height={340}>
            {/* modelComparisonGroupedBarChart: Compare accuracy, precision, recall, and F1-score by model. */}
            <BarChart data={chartData} barGap={4}>
              <CartesianGrid vertical={false} stroke="rgba(148, 163, 184, 0.14)" />
              <XAxis dataKey="name" tickLine={false} axisLine={false} />
              <YAxis tickFormatter={(value) => `${value}%`} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTooltip formatter={(value) => `${Number(value).toFixed(1)}%`} />} />
              <Bar dataKey="accuracy" name="Accuracy" radius={[6, 6, 0, 0]}>
                {chartData.map((item) => (
                  <Cell key={item.name} fill={METRIC_COLORS.accuracy} fillOpacity={item.name === bestModel ? 1 : 0.55} />
                ))}
              </Bar>
              <Bar dataKey="precision" name="Precision" radius={[6, 6, 0, 0]}>
                {chartData.map((item) => (
                  <Cell key={item.name} fill={METRIC_COLORS.precision} fillOpacity={item.name === bestModel ? 1 : 0.55} />
                ))}
              </Bar>
              <Bar dataKey="recall" name="Recall" radius={[6, 6, 0, 0]}>
                {chartData.map((item) => (
                  <Cell key={item.name} fill={METRIC_COLORS.recall} fillOpacity={item.name === bestModel ? 1 : 0.55} />
                ))}
              </Bar>
              <Bar dataKey="f1" name="F1-score" radius={[6, 6, 0, 0]}>
                {chartData.map((item) => (
                  <Cell key={item.name} fill={METRIC_COLORS.f1} fillOpacity={item.name === bestModel ? 1 : 0.55} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="metric-note">
          This chart compares how dependable each model is at reading customer emotions. Higher bars mean the model is more consistent.
          SVM is currently the best choice because it gives the strongest balance of correct results and fewer missed emotion signals,
          so Sentimine uses it as the fixed model for sentence and dataset classification.
        </p>
      </div>
    </section>
  );
}

// ============================================
// PROCESS PAGE
// ============================================

// Documentation page explaining single sentence and dataset classification flows
function ProcessPage() {
  return (
    <div className="page-stack">
      <section className="page-intro">
        <p className="eyebrow">Classification process</p>
        <h1>From text input to explainable emotion output</h1>
        <p>
          Sentimine keeps the flow transparent: text is transformed, classified, explained, and summarized into results
          that are easy to inspect during a presentation.
        </p>
      </section>
      <FlowSection
        title="Single Sentence Flow"
        steps={[
          ["Input", Search, "Enter a sentence for classification.", "The sentence is trimmed and sent to the backend as one review-sized text sample.", ["Validate non-empty text", "Preserve original sentence for display"]],
          ["Vectorize (TF-IDF)", Layers3, "Convert text into weighted features.", "Terms that matter more to the model receive stronger weights while common words contribute less.", ["Clean and tokenize text", "Apply saved TF-IDF vocabulary", "Create sparse feature vector"]],
          ["Model Inference", Brain, "Run the best trained classifier.", "The trained emotion model predicts happy, sad, or angry and returns probability scores.", ["Use fixed best model", "Generate class prediction", "Return happy/sad/angry probabilities"]],
          ["Confidence Scoring", Sparkles, "Calculate emotion certainty.", "The platform returns happy, sad, and angry percentages so the presenter can explain the model's confidence.", ["Normalize probability scores", "Find the dominant emotion", "Prepare percentage table"]],
          ["Output", CheckCircle2, "Return emotion and confidence.", "The dominant emotion emoji and percentage table appear in the hero result panel.", ["Show dominant emoji", "Show confidence percentages", "Enable Try Another reset"]],
        ]}
        detail="Emotion plus confidence breakdown"
      />
      <FlowSection
        title="Dataset (CSV/Excel) Flow"
        steps={[
          ["Upload CSV/Excel", Upload, "Select a review dataset.", "A file can be dropped into the upload zone or selected from the browser picker.", ["Download template with fixed headers", "Accept drag-and-drop or click upload", "Keep upload, reset, and template actions visible"]],
          ["Parse and Validate", FileSpreadsheet, "Check file rows and columns.", "The platform reads Category, Product Name, and Review fields, then falls back locally when the API is unavailable.", ["Browser File API reads the file", "CSV parser maps fixed headers", "Inline errors explain unreadable files"]],
          ["Batch Classify", Brain, "Classify reviews in batches.", "Rows are classified through the saved model when the backend is running, or through the offline fallback for presentation mode.", ["Vectorize each review through backend model", "Use offline keyword fallback when needed", "Attach emotion result per row"]],
          ["Aggregate Results", Layers3, "Summarize emotion patterns.", "Counts, top products, top categories, and word-cloud terms are computed from classified rows.", ["Count happy/sad/angry totals", "Rank products/categories by selected emotion", "Extract emotion-specific top words"]],
          ["Dashboard Output", CheckCircle2, "Render KPIs and charts.", "Visualizations remain visible in zero, original, uploaded, and offline states.", ["Recharts: KPI charts, bars, donut, grouped bars", "d3-cloud: compact deterministic word cloud", "Browser Print API: export dashboard as PDF", "Lucide React: upload, file, flow, and action icons"]],
        ]}
      />
      <section className="section library-section">
        <div className="section-heading">
          <p className="eyebrow">Frontend libraries and utilities</p>
          <h2>What powers the live dashboard experience</h2>
        </div>
        <div className="library-grid">
          {[
            ["Recharts", "Draws the emotion count chart, donut share chart, grouped product/category chart, and model comparison bars."],
            ["d3-cloud", "Places word-cloud terms into a compact, repeatable layout without tilted or random-looking words."],
            ["Browser File API", "Supports drag-and-drop upload, click upload, and local CSV reading for offline presentations."],
            ["Browser Print API", "Exports the dataset dashboard section as a printable PDF with the current visualizations."],
            ["Lucide React", "Provides clean icons for upload, dataset, classification, and process steps."],
            ["Offline fallback service", "Keeps sentence classification, dataset summaries, and model comparison working when the API is not running."],
          ].map(([name, purpose]) => (
            <div className="library-card" key={name}>
              <strong>{name}</strong>
              <p>{purpose}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// ============================================
// TRAINING PAGE
// ============================================

// Documentation page explaining model training pipeline and libraries
function TrainingPage() {
  return (
    <div className="page-stack">
      <section className="page-intro">
        <p className="eyebrow">Model training</p>
        <h1>Training pipeline for the filtered review dataset</h1>
        <p>
          The default training source is datasets/filtered_dataset.csv, using category, product name, emotion, and review
          as the core fields for emotion classification.
        </p>
      </section>
      <section className="section dataset-schema">
        <div className="schema-grid">
          {["category", "product name", "emotion", "review"].map((column) => (
            <span key={column}>{column}</span>
          ))}
        </div>
      </section>
      <section className="section library-section">
        <div className="section-heading">
          <p className="eyebrow">Training libraries</p>
          <h2>Libraries used to prepare, train, and evaluate the emotion model</h2>
        </div>
        <div className="library-grid">
          {[
            ["pandas", "Loads the CSV dataset, cleans rows, groups products/categories, and prepares review text."],
            ["NumPy", "Handles numeric arrays and probability values used during model comparison."],
            ["scikit-learn", "Provides TF-IDF vectorization, train/test splitting, classifiers, and evaluation metrics."],
            ["TF-IDF Vectorizer", "Turns review words into weighted numeric features that machine learning models can understand."],
            ["Logistic Regression", "Fast baseline model used to compare against the final selected classifier."],
            ["Naive Bayes", "Lightweight text classifier that works well as a simple benchmark."],
            ["Linear SVM", "Best fixed model for this platform; strong for text classification and emotion separation."],
            ["Random Forest", "Tree-based comparison model used to check how non-linear learners perform on the dataset."],
            ["d3-cloud", "Used on the frontend to render the compact emotion word cloud after dataset results are aggregated."],
            ["Recharts", "Used on the frontend to present training comparison and dataset dashboard charts."],
            ["Browser File API", "Lets the presentation version read uploaded CSV files locally when the backend is unavailable."],
            ["Browser Print API", "Exports the current dataset dashboard as a PDF-ready report for stakeholders."],
          ].map(([name, purpose]) => (
            <div className="library-card" key={name}>
              <strong>{name}</strong>
              <p>{purpose}</p>
            </div>
          ))}
        </div>
      </section>
        <FlowSection
          title="Training Pipeline"
          steps={[
            ["Raw Dataset", FileSpreadsheet, "Start with labeled review rows.", "The filtered dataset contains only the emotion labels used by the platform: happy, sad, and angry.", ["Load datasets/filtered_dataset.csv", "Use category, product name, emotion, review"]],
            ["Data Cleaning", Sparkles, "Normalize text and remove noise.", "Reviews are cleaned so inconsistent casing, symbols, and extra spacing do not distract the model.", ["Lowercase text", "Remove noisy symbols", "Drop empty reviews"]],
            ["Preprocessing", Layers3, "Tokenize, remove stopwords, vectorize.", "Text becomes TF-IDF features that represent meaningful terms as numeric input.", ["Tokenize review text", "Remove common stop words", "Build TF-IDF matrix"]],
            ["Train/Test Split", ArrowRight, "Create reliable evaluation data.", "A held-out test split checks whether the model generalizes beyond the training examples.", ["80/20 split", "Fixed random seed", "Same split for all models"]],
            ["Model Training", Brain, "Fit SVM, Logistic Regression, Random Forest, and Naive Bayes.", "Multiple algorithms train on the same features so their performance can be compared fairly.", ["Train four classifiers", "Use class balancing where helpful", "Keep trained candidates in memory"]],
            ["Evaluation", CheckCircle2, "Compare accuracy, precision, recall, and F1.", "The best model is selected by F1-score because it balances precision and recall.", ["Calculate weighted metrics", "Compare model table", "Pick highest F1-score"]],
            ["Save Model and Vectorizer", Upload, "Persist the best pipeline.", "The trained classifier and vectorizer are reused for sentence and dataset classification.", ["Save best_model.pkl", "Reuse fixed model for uploads", "Serve comparison metrics to frontend"]],
          ]}
          detail="SVM, Logistic Regression, Random Forest, Naive Bayes"
        />
    </div>
  );
}

// ============================================
// FLOW SECTION & HELPERS
// ============================================

// Renders a multi-step flow diagram with step cards and detailed information
function FlowSection({ title, steps, detail }) {
  return (
    <section className="section flow-section">
      <div className="section-heading">
        <p className="eyebrow">{title}</p>
        {detail && <h2>{detail}</h2>}
      </div>
      <div className="flow-grid">
        {steps.map(([label, icon, description, detailText, substeps], index) => {
          const StepIcon = icon;

          return (
            <div className="flow-node" key={label}>
              <span className="step-number">{String(index + 1).padStart(2, "0")}</span>
              <div className="step-card">
                <span className="step-icon"><StepIcon size={22} /></span>
                <strong>{label}</strong>
                {description && <p>{description}</p>}
                {detailText && <small>{detailText}</small>}
                {substeps?.length > 0 && (
                  <ul className="substep-list">
                    {substeps.map((substep) => <li key={substep}>{substep}</li>)}
                  </ul>
                )}
              </div>
              {index < steps.length - 1 && <ArrowRight className="flow-arrow" size={22} />}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ============================================
// UTILITY HELPER FUNCTIONS
// ============================================

// Capitalize first letter of a string
function capitalize(value) {
  return value ? `${value.charAt(0).toUpperCase()}${value.slice(1)}` : "";
}

// Seeded random number generator for deterministic word cloud layout
function seededRandom(seed) {
  let value = seed % 2147483647;
  if (value <= 0) value += 2147483646;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}


// Normalize product/category data structure from API response
function normalizeTopItems(items = []) {
  return items.map((item, index) => {
    const name = item.name || item.product || item.category || item.label || `Item ${index + 1}`;
    return {
      name: name.length > 20 ? `${name.slice(0, 18)}...` : name,
      happy: Number(item.happy || item.happy_count || item.counts?.happy || 0),
      sad: Number(item.sad || item.sad_count || item.counts?.sad || 0),
      angry: Number(item.angry || item.angry_count || item.counts?.angry || 0),
    };
  });
}

// Normalize word cloud data structure from API response
function normalizeWords(words = []) {
  return words.map((item) => ({
    word: item.word,
    count: Number(item.count || item.value || 0),
    emotion: normalizeEmotion(item.emotion || item.label || "happy"),
  }));
}

// ============================================
// MAIN APP COMPONENT
// ============================================

// Root component managing page navigation between Home, Process, and Training tabs
export default function App() {
  const [page, setPage] = useState("home");

  return (
    <div className="app">
      <Header tabs={tabs} activeTab={page} onChange={setPage} />
      <main className="main">
        {page === "home" && <HomePage />}
        {page === "process" && <ProcessPage />}
        {page === "training" && <TrainingPage />}
      </main>
    </div>
  );
}
// categoryEmotionHeatmap: Display frequency of emotions per category in a grid format (Emotions=Vertical, Categories=Horizontal).
function CategoryEmotionHeatmap({ data }) {
  const categories = Object.keys(data || {});
  if (categories.length === 0) return <div className="muted-center">No category data available for heatmap</div>;
  const maxVal = Math.max(...categories.flatMap(cat => Object.values(data[cat])), 1);

  return (
    <div className="heatmap-wrapper">
      <div className="heatmap-scroll-box">
        <div 
          className="heatmap-grid" 
          style={{ gridTemplateColumns: `80px repeat(${categories.length}, 80px)` }}
        >
          {/* Header Row: Spacer + Categories */}
          <div className="heatmap-corner"></div>
          {categories.map(cat => (
            <div key={cat} className="heatmap-col-label" title={cat}>{cat}</div>
          ))}

          {/* Emotion Rows */}
          {EMOTION_KEYS.map(key => (
            <React.Fragment key={key}>
              <div className="heatmap-row-label">{EMOTION_LABELS[key]}</div>
              {categories.map(cat => {
                const val = data[cat][key] || 0;
                const intensity = val / maxVal;
                return (
                  <div 
                    key={cat} 
                    className="heatmap-cell" 
                    style={{ 
                      backgroundColor: `${EMOTION_COLORS[key]}${Math.round((0.1 + intensity * 0.9) * 255).toString(16).padStart(2, '0')}`,
                    }}
                    title={`${EMOTION_LABELS[key]} in ${cat}: ${val}`}
                  >
                    <span className="heatmap-cell-val">{val > 0 ? val : ""}</span>
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

// reviewLengthBoxPlot: Display word count distribution per emotion using a box plot visualization.
function ReviewLengthBoxPlot({ data }) {
  const hasLens = data && Object.values(data).some(d => d.values && d.values.length > 0);
  if (!hasLens) return <div className="muted-center">No review length data available</div>;

  return (
    <div className="boxplot-container">
      {EMOTION_KEYS.map(key => {
        const stats = data[key] || { min: 0, max: 0, median: 0, mean: 0, values: [] };
        const q1 = stats.values[Math.floor(stats.values.length * 0.25)] || 0;
        const q3 = stats.values[Math.floor(stats.values.length * 0.75)] || 0;
        const globalMax = Math.max(...EMOTION_KEYS.map(k => data[k]?.max || 0), 1);
        
        const scale = (val) => (val / globalMax) * 100;

        return (
          <div key={key} className="boxplot-row">
            <div className="boxplot-label">{EMOTION_LABELS[key]}</div>
            <div className="boxplot-track">
              {/* Whisker line */}
              <div className="boxplot-whisker" style={{ left: `${scale(stats.min)}%`, width: `${scale(stats.max - stats.min)}%` }}></div>
              {/* Box (Q1 to Q3) */}
              <div 
                className="boxplot-box" 
                style={{ 
                  left: `${scale(q1)}%`, 
                  width: `${scale(q3 - q1)}%`,
                  backgroundColor: EMOTION_COLORS[key]
                }}
              ></div>
              {/* Median Line */}
              <div className="boxplot-median" style={{ left: `${scale(stats.median)}%` }}></div>
            </div>
            <div className="boxplot-stats">{Math.round(stats.median)} words</div>
          </div>
        );
      })}
      <div className="boxplot-axis">
        <span>0</span>
        <span>Word Count</span>
        <span>{Math.max(...EMOTION_KEYS.map(k => data[k]?.max || 0))}</span>
      </div>
    </div>
  );
}

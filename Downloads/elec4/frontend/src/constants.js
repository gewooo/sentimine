export const EMOTION_COLORS = {
  happy: "#f2b705",
  sad: "#2f80ed",
  angry: "#e23d3d",
};

export const EMOTION_LABELS = {
  happy: "happy",
  sad: "sad",
  angry: "angry",
};

export const EMOTION_EMOJIS = {
  happy: "😊",
  sad: "😢",
  angry: "😡",
};

export const EMOTION_KEYS = ["happy", "sad", "angry"];

export const METRIC_COLORS = {
  accuracy: "#2357d8",
  precision: "#13a085",
  recall: "#7c5cc4",
  f1: "#f2b705",
};

export function normalizeEmotion(value = "") {
  const emotion = String(value).trim().toLowerCase();
  if (emotion === "sadness") return "sad";
  if (emotion === "anger") return "angry";
  if (EMOTION_KEYS.includes(emotion)) return emotion;
  return "happy";
}

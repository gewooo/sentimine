import "./Header.css";

export default function Header({ tabs, activeTab, onChange }) {
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <div className="wordmark" aria-label="sentimine Emotion Classification Platform">
          <strong>sentimine</strong>
          <span>Emotion Classification Platform</span>
        </div>

        <nav className="top-nav" aria-label="Primary navigation">
          {Object.entries(tabs).map(([key, label]) => (
            <button key={key} className={activeTab === key ? "active" : ""} onClick={() => onChange(key)}>
              {label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}

/** Inline DEEP-Dari mark — teal D + amber Persian curve; follows theme colors. */
export default function DeepDariLogo({ className = "brand-logo", title = "DEEP-Dari" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 64 64"
      width="34"
      height="34"
      role="img"
      aria-label={title}
      xmlns="http://www.w3.org/2000/svg"
    >
      <title>{title}</title>
      <defs>
        <linearGradient id="dd-d" x1="12" y1="8" x2="52" y2="56" gradientUnits="userSpaceOnUse">
          <stop stopColor="var(--logo-teal)" />
          <stop offset="1" stopColor="var(--logo-teal-deep)" />
        </linearGradient>
      </defs>
      {/* Soft plate */}
      <rect x="2" y="2" width="60" height="60" rx="14" fill="var(--logo-plate)" />
      {/* Stylized D */}
      <path
        fill="url(#dd-d)"
        d="M18 14h14c11.5 0 20 8.2 20 18s-8.5 18-20 18H18V14zm8 8v20h6c6.6 0 12-5.4 12-10s-5.4-10-12-10h-6z"
      />
      {/* Persian-inspired curve (emotion / language) */}
      <path
        fill="none"
        stroke="var(--logo-amber)"
        strokeWidth="3.2"
        strokeLinecap="round"
        d="M22 46c6-10 14-14 24-12 4 .8 8 3.2 11 6.5"
      />
      <circle cx="47" cy="32" r="3.2" fill="var(--logo-amber)" />
    </svg>
  );
}

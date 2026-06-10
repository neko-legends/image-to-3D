/// <reference types="vite/client" />

declare namespace JSX {
  interface IntrinsicElements {
    "model-viewer": React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
      src?: string;
      "camera-controls"?: boolean;
      "auto-rotate"?: boolean;
      "rotation-per-second"?: string;
      "shadow-intensity"?: string;
      exposure?: string;
    };
  }
}

import type { Metadata } from "next";

import "@copilotkit/react-core/v2/styles.css";
import "./globals.css";

import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "financial-agent",
  description: "Portfolio and signal analysis over E-Trade data",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

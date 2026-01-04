import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

export const metadata: Metadata = {
  title: "Kiosk Agent",
  description: "Android Kiosk Control Agent Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <CopilotKit runtimeUrl="http://localhost:8000/copilotkit">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}


import type { Metadata, Viewport } from "next";
import { Space_Grotesk } from "next/font/google";
import Script from "next/script";
import { Analytics } from "@vercel/analytics/next";

import { getSiteUrl } from "@/lib/site";
import { APP_THEME_COLOR, createMetadata } from "@/web-seo-metadata";

import "./globals.css";

const DEFAULT_GA_MEASUREMENT_ID = "G-JWWWXJQQEP";

const sans = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = createMetadata(getSiteUrl());

export const viewport: Viewport = {
  themeColor: APP_THEME_COLOR,
};

function getGoogleAnalyticsId(): string {
  return (
    process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID?.trim() ||
    DEFAULT_GA_MEASUREMENT_ID
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const gaMeasurementId = getGoogleAnalyticsId();

  return (
    <html lang="en">
      <body className={sans.variable}>
        <Script
          src={`https://www.googletagmanager.com/gtag/js?id=${gaMeasurementId}`}
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag("js", new Date());
            gtag("config", "${gaMeasurementId}");
          `}
        </Script>
        {children}
        <Analytics />
      </body>
    </html>
  );
}

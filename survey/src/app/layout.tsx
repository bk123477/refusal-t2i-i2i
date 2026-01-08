import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'I2I Refusal Evaluation',
  description: 'Human evaluation survey for I2I refusal bias study',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-gray-900 text-white">{children}</body>
    </html>
  )
}

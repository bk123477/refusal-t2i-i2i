import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ACRB Human Evaluation Survey',
  description: 'Human evaluation survey for T2I/I2I model refusal bias assessment',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <main className="min-h-screen py-8 px-4">
          {children}
        </main>
      </body>
    </html>
  )
}

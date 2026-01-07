'use client'

import Image from 'next/image'
import { useState } from 'react'

interface ImageDisplayProps {
  src: string
  alt: string
  prompt?: string
  isRefusal?: boolean
  refusalMessage?: string
}

export default function ImageDisplay({
  src,
  alt,
  prompt,
  isRefusal = false,
  refusalMessage,
}: ImageDisplayProps) {
  const [isLoading, setIsLoading] = useState(true)
  const [hasError, setHasError] = useState(false)

  if (isRefusal && refusalMessage) {
    return (
      <div className="w-full">
        {prompt && (
          <div className="mb-4 border-l-2 border-neutral-200 pl-4 py-1">
            <span className="text-xs text-neutral-500 uppercase tracking-widest">Prompt</span>
            <p className="text-sm text-neutral-700 mt-1">{prompt}</p>
          </div>
        )}
        <div className="aspect-square bg-neutral-100 border border-neutral-200 flex items-center justify-center p-8">
          <div className="text-center">
            <div className="w-12 h-12 border border-neutral-300 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-neutral-400 text-xl">×</span>
            </div>
            <p className="text-neutral-900 text-sm font-medium">Request Blocked</p>
            <p className="text-neutral-500 text-xs mt-2">{refusalMessage}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      {prompt && (
        <div className="mb-4 border-l-2 border-neutral-200 pl-4 py-1">
          <span className="text-xs text-neutral-500 uppercase tracking-widest">Prompt</span>
          <p className="text-sm text-neutral-700 mt-1">{prompt}</p>
        </div>
      )}
      <div className="relative aspect-square bg-neutral-100 border border-neutral-200 overflow-hidden">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-6 h-6 border border-neutral-300 border-t-neutral-900 rounded-full animate-spin" />
          </div>
        )}
        {hasError ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-neutral-400">
              <p className="text-sm">Failed to load</p>
            </div>
          </div>
        ) : (
          <Image
            src={src}
            alt={alt}
            fill
            className={`object-contain transition-opacity duration-300 ${
              isLoading ? 'opacity-0' : 'opacity-100'
            }`}
            onLoad={() => setIsLoading(false)}
            onError={() => {
              setIsLoading(false)
              setHasError(true)
            }}
          />
        )}
      </div>
    </div>
  )
}

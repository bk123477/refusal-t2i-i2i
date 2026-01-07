'use client'

interface ProgressBarProps {
  current: number
  total: number
}

export default function ProgressBar({ current, total }: ProgressBarProps) {
  const percentage = Math.round((current / total) * 100)

  return (
    <div className="mb-6">
      <div className="flex justify-between text-xs text-neutral-400 mb-2 uppercase tracking-widest">
        <span>Progress</span>
        <span>{percentage}%</span>
      </div>
      <div className="w-full h-px bg-neutral-200 relative">
        <div
          className="absolute left-0 top-0 h-px bg-neutral-900 transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

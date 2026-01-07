'use client'

interface LikertScaleProps {
  value: number | null
  onChange: (value: number) => void
  labels?: string[]
  min?: number
  max?: number
}

export default function LikertScale({
  value,
  onChange,
  labels = ['Very Poor', 'Poor', 'Neutral', 'Good', 'Excellent'],
  min = 1,
  max = 5,
}: LikertScaleProps) {
  const options = Array.from({ length: max - min + 1 }, (_, i) => min + i)

  return (
    <div className="flex justify-center gap-3">
      {options.map((option, index) => (
        <button
          key={option}
          onClick={() => onChange(option)}
          className={`flex flex-col items-center gap-2 p-4 min-w-[72px] border transition-all ${
            value === option
              ? 'border-neutral-900 bg-neutral-50'
              : 'border-neutral-200 hover:border-neutral-400'
          }`}
        >
          <span className="text-lg font-light text-neutral-900">{option}</span>
          <span className="text-[10px] text-neutral-500 uppercase tracking-wider">
            {labels[index] || ''}
          </span>
        </button>
      ))}
    </div>
  )
}

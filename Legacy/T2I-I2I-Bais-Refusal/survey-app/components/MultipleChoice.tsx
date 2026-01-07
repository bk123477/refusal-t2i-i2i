'use client'

interface Option {
  value: string
  label: string
  description?: string
}

interface MultipleChoiceProps {
  options: Option[]
  value: string | null
  onChange: (value: string) => void
  columns?: 1 | 2 | 4
}

export default function MultipleChoice({
  options,
  value,
  onChange,
  columns = 2,
}: MultipleChoiceProps) {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    4: 'grid-cols-2 sm:grid-cols-4',
  }

  return (
    <div className={`grid ${gridCols[columns]} gap-2`}>
      {options.map((option) => (
        <button
          key={option.value}
          onClick={() => onChange(option.value)}
          className={`p-4 border text-left transition-all ${
            value === option.value
              ? 'border-neutral-900 bg-neutral-50'
              : 'border-neutral-200 hover:border-neutral-400'
          }`}
        >
          <div className="flex items-center gap-3">
            <div
              className={`w-4 h-4 rounded-full border flex-shrink-0 flex items-center justify-center ${
                value === option.value
                  ? 'border-neutral-900 bg-neutral-900'
                  : 'border-neutral-300'
              }`}
            >
              {value === option.value && (
                <div className="w-1.5 h-1.5 rounded-full bg-white" />
              )}
            </div>
            <div>
              <p className="text-sm text-neutral-900">{option.label}</p>
              {option.description && (
                <p className="text-xs text-neutral-500 mt-0.5">{option.description}</p>
              )}
            </div>
          </div>
        </button>
      ))}
    </div>
  )
}

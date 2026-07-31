import React from 'react'

function ImageUploader({ onFileSelect, previewUrl, previewLabel, loading }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
      <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-12 text-center transition hover:border-emerald-500 hover:bg-emerald-50">
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => onFileSelect(event.target.files?.[0])}
        />
        <div className="rounded-full bg-emerald-100 p-3 text-emerald-700">⬆</div>
        <div>
          <p className="text-lg font-semibold">Drag and drop or click to upload</p>
          <p className="mt-1 text-sm text-slate-500">Supports JPG, PNG, and WEBP files.</p>
        </div>
      </label>

      <div className="mt-4 text-sm text-slate-500">{previewLabel}</div>

      {previewUrl ? (
        <div className="mt-5 overflow-hidden rounded-2xl border border-slate-200 bg-white">
          <img src={previewUrl} alt="Selected leaf preview" className="max-h-96 w-full object-contain" />
        </div>
      ) : null}
    </div>
  )
}

export default ImageUploader

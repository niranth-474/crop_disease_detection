import React from 'react'

function Loader() {
  return (
    <div className="flex items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 p-10">
      <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-200 border-t-emerald-600" />
    </div>
  )
}

export default Loader

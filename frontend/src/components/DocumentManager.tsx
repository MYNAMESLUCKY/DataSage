import React from 'react';

interface DocumentManagerProps {
  user: any;
  className?: string;
}

export default function DocumentManager({ user, className }: DocumentManagerProps) {
  return (
    <div className={className}>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Document Manager</h2>
        
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <div className="text-gray-500 mb-2">
              <svg className="mx-auto h-12 w-12" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-1">Upload Documents</h3>
            <p className="text-gray-500">Drag and drop files here, or click to browse</p>
            <p className="text-sm text-gray-400 mt-2">Supports PDF, DOC, TXT, CSV, and Excel files</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Recent Uploads</h4>
              <p className="text-sm text-gray-600">No documents uploaded yet</p>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Processing Status</h4>
              <p className="text-sm text-gray-600">Ready to process documents</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
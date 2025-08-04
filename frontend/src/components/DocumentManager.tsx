import React, { useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface DocumentManagerProps {
  user: any;
}

export default function DocumentManager({ user }: DocumentManagerProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [documents] = useState([
    { id: 1, name: 'company-handbook.pdf', size: '2.3 MB', uploaded: '2024-08-03', status: 'processed' },
    { id: 2, name: 'technical-specs.docx', size: '1.1 MB', uploaded: '2024-08-02', status: 'processing' },
    { id: 3, name: 'user-manual.txt', size: '456 KB', uploaded: '2024-08-01', status: 'processed' }
  ]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      setIsUploading(true);
      // Simulate upload
      setTimeout(() => {
        setIsUploading(false);
        alert('File uploaded successfully! Processing will begin shortly.');
      }, 2000);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h2 className="text-xl font-bold mb-4 text-gray-900">Document Management</h2>
        <p className="text-gray-600 mb-6">
          Upload and manage documents for your RAG knowledge base
        </p>

        {/* Upload Section */}
        <div className="card" style={{ backgroundColor: '#f8fafc', marginBottom: '24px' }}>
          <h3 className="text-lg font-semibold mb-3 text-gray-900">Upload New Document</h3>
          
          <div style={{ 
            border: '2px dashed #d1d5db', 
            borderRadius: '8px', 
            padding: '40px 20px', 
            textAlign: 'center',
            backgroundColor: 'white'
          }}>
            {isUploading ? (
              <div>
                <LoadingSpinner size="lg" />
                <p className="text-gray-600 mt-4">Uploading document...</p>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>📄</div>
                <p className="text-gray-600 mb-4">
                  Drag and drop files here, or click to browse
                </p>
                <input
                  type="file"
                  onChange={handleFileUpload}
                  accept=".pdf,.docx,.txt,.md"
                  style={{ display: 'none' }}
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="button">
                  Choose Files
                </label>
                <p style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px' }}>
                  Supported: PDF, DOCX, TXT, MD (Max: 10MB for free tier)
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Documents List */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 text-gray-900">Your Documents</h3>
          
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <th style={{ padding: '12px', textAlign: 'left', color: '#6b7280', fontWeight: '500' }}>Document</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: '#6b7280', fontWeight: '500' }}>Size</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: '#6b7280', fontWeight: '500' }}>Uploaded</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: '#6b7280', fontWeight: '500' }}>Status</th>
                  <th style={{ padding: '12px', textAlign: 'left', color: '#6b7280', fontWeight: '500' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map(doc => (
                  <tr key={doc.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '12px' }}>
                      <div className="flex items-center">
                        <span style={{ marginRight: '8px' }}>📄</span>
                        <span className="font-semibold text-gray-900">{doc.name}</span>
                      </div>
                    </td>
                    <td style={{ padding: '12px', color: '#6b7280' }}>{doc.size}</td>
                    <td style={{ padding: '12px', color: '#6b7280' }}>{doc.uploaded}</td>
                    <td style={{ padding: '12px' }}>
                      <span style={{
                        padding: '4px 8px',
                        borderRadius: '12px',
                        fontSize: '12px',
                        fontWeight: '500',
                        backgroundColor: doc.status === 'processed' ? '#dcfce7' : '#fef3c7',
                        color: doc.status === 'processed' ? '#166534' : '#92400e'
                      }}>
                        {doc.status}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>
                      <button
                        className="button"
                        style={{ 
                          padding: '6px 12px', 
                          fontSize: '12px',
                          backgroundColor: '#6b7280',
                          marginRight: '8px'
                        }}
                      >
                        View
                      </button>
                      <button
                        className="button"
                        style={{ 
                          padding: '6px 12px', 
                          fontSize: '12px',
                          backgroundColor: '#dc2626'
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Storage Usage */}
        <div className="card" style={{ backgroundColor: '#f8fafc' }}>
          <h3 className="text-lg font-semibold mb-3 text-gray-900">Storage Usage</h3>
          <div style={{ marginBottom: '8px' }}>
            <div className="flex justify-between">
              <span className="text-gray-600">Used: 3.9 MB</span>
              <span className="text-gray-600">Limit: 10 MB (Free Tier)</span>
            </div>
          </div>
          <div style={{ 
            width: '100%', 
            height: '8px', 
            backgroundColor: '#e5e7eb', 
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <div style={{ 
              width: '39%', 
              height: '100%', 
              backgroundColor: '#3b82f6',
              borderRadius: '4px'
            }}></div>
          </div>
          <p style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px' }}>
            Upgrade to Pro for 100 MB storage limit
          </p>
        </div>
      </div>
    </div>
  );
}
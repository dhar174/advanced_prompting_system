import React, { useState } from 'react';
import type { OutputTypeInfo } from '../graphql/graphqlTypes';

export interface StructuredOutputRendererProps {
  content: string;
  outputType?: OutputTypeInfo | null;
}

export const StructuredOutputRenderer: React.FC<StructuredOutputRendererProps> = ({
  content,
  outputType = { outputType: 'Simple Concise Answer', fileExtension: '.txt' },
}) => {
  const [copied, setCopied] = useState(false);
  const [showRawCsv, setShowRawCsv] = useState(false);

  const typeName = outputType?.outputType || 'Simple Concise Answer';
  const fileExt = outputType?.fileExtension || '.txt';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  const isCode =
    ['Python Script', 'Code Snippet', 'JSON', 'HTML', 'CSV'].includes(typeName) ||
    ['.py', '.json', '.html', '.csv', '.js', '.ts', '.sh'].includes(fileExt);

  // Pretty-print JSON if applicable
  let formattedJson: string | null = null;
  if (typeName === 'JSON' || fileExt === '.json') {
    try {
      formattedJson = JSON.stringify(JSON.parse(content), null, 2);
    } catch {
      formattedJson = null;
    }
  }

  // Parse CSV if applicable
  const isCsv = typeName === 'CSV' || fileExt === '.csv';
  let csvRows: string[][] | null = null;
  if (isCsv) {
    try {
      const lines = content.trim().split('\n');
      if (lines.length > 0 && lines[0].includes(',')) {
        csvRows = lines.map((l) => l.split(',').map((cell) => cell.trim()));
      }
    } catch {
      csvRows = null;
    }
  }

  return (
    <div className="mt-2 space-y-2">
      {/* Metadata & Actions Header */}
      <div className="flex items-center justify-between text-xs text-slate-500 bg-slate-50 px-3 py-1.5 rounded border border-slate-200">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-slate-700">{typeName}</span>
          <span className="bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded font-mono">
            {fileExt}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          {isCsv && csvRows && (
            <button
              type="button"
              onClick={() => setShowRawCsv(!showRawCsv)}
              className="text-xs text-blue-600 hover:text-blue-800 underline"
            >
              {showRawCsv ? 'Show Table' : 'Show Raw CSV'}
            </button>
          )}
          <button
            type="button"
            onClick={handleCopy}
            className="px-2 py-0.5 bg-white border border-slate-300 rounded text-xs text-slate-700 hover:bg-slate-100 transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      {/* Rendered Body */}
      {isCode ? (
        isCsv && csvRows && !showRawCsv ? (
          <div className="overflow-x-auto border border-slate-200 rounded-md">
            <table className="min-w-full text-xs divide-y divide-slate-200">
              <thead className="bg-slate-100">
                <tr>
                  {csvRows[0].map((header, idx) => (
                    <th
                      key={idx}
                      className="px-3 py-2 text-left font-semibold text-slate-700"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 bg-white">
                {csvRows.slice(1).map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-slate-50">
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} className="px-3 py-1.5 text-slate-700">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <pre className="p-3 bg-slate-900 text-slate-100 rounded-md overflow-x-auto text-xs font-mono leading-relaxed border border-slate-800">
            <code data-testid="structured-code-output">
              {formattedJson || content}
            </code>
          </pre>
        )
      ) : (
        <div className="p-3 bg-green-100/50 border border-green-300 rounded-md text-sm text-green-800 whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  );
};

export default StructuredOutputRenderer;

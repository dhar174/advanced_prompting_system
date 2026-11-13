/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import StructuredOutputRenderer from './StructuredOutputRenderer';

describe('StructuredOutputRenderer', () => {
  it('renders simple concise answer text cleanly', () => {
    render(
      <StructuredOutputRenderer
        content="Here is the simple concise answer."
        outputType={{ outputType: 'Simple Concise Answer', fileExtension: '.txt' }}
      />
    );
    expect(screen.getByText('Simple Concise Answer')).toBeInTheDocument();
    expect(screen.getByText('.txt')).toBeInTheDocument();
    expect(screen.getByText('Here is the simple concise answer.')).toBeInTheDocument();
  });

  it('renders syntax-formatted code container for Python Script', () => {
    const pythonCode = 'def hello():\n    return "world"';
    render(
      <StructuredOutputRenderer
        content={pythonCode}
        outputType={{ outputType: 'Python Script', fileExtension: '.py' }}
      />
    );
    expect(screen.getByText('Python Script')).toBeInTheDocument();
    expect(screen.getByText('.py')).toBeInTheDocument();
    const codeElem = screen.getByTestId('structured-code-output');
    expect(codeElem).toBeInTheDocument();
    expect(codeElem.textContent).toContain('def hello():');
  });

  it('pretty-prints JSON content when valid JSON is provided', () => {
    const jsonStr = '{"key":"value","count":42}';
    render(
      <StructuredOutputRenderer
        content={jsonStr}
        outputType={{ outputType: 'JSON', fileExtension: '.json' }}
      />
    );
    expect(screen.getByText('JSON')).toBeInTheDocument();
    const codeElem = screen.getByTestId('structured-code-output');
    expect(codeElem.textContent).toContain('"key": "value"');
    expect(codeElem.textContent).toContain('"count": 42');
  });

  it('renders tabular preview for CSV content and allows toggling raw view', () => {
    const csvContent = 'name,role,level\nAlice,Developer,Senior\nBob,Designer,Mid';
    render(
      <StructuredOutputRenderer
        content={csvContent}
        outputType={{ outputType: 'CSV', fileExtension: '.csv' }}
      />
    );
    expect(screen.getByText('CSV')).toBeInTheDocument();
    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Developer')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();

    const toggleBtn = screen.getByRole('button', { name: /Show Raw CSV/i });
    fireEvent.click(toggleBtn);
    expect(screen.getByTestId('structured-code-output')).toBeInTheDocument();
  });

  it('copies content to clipboard on copy button click', async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    render(
      <StructuredOutputRenderer
        content="Copyable text content"
        outputType={{ outputType: 'Text File', fileExtension: '.txt' }}
      />
    );

    const copyBtn = screen.getByRole('button', { name: /Copy/i });
    fireEvent.click(copyBtn);
    expect(writeTextMock).toHaveBeenCalledWith('Copyable text content');
  });
});

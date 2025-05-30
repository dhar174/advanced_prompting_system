/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen }
from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from './Card'; // Assuming components are exported from './Card'

describe('Card Components', () => {
  describe('Card', () => {
    it('renders children correctly', () => {
      render(<Card>Child content</Card>);
      expect(screen.getByText('Child content')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      render(<Card className="custom-card">Child content</Card>);
      expect(screen.getByText('Child content')).toHaveClass('custom-card');
    });

    it('spreads other props like id and aria-label', () => {
      render(
        <Card id="my-card" aria-label="A simple card">
          Child content
        </Card>
      );
      const cardElement = screen.getByText('Child content');
      expect(cardElement).toHaveAttribute('id', 'my-card');
      expect(cardElement).toHaveAttribute('aria-label', 'A simple card');
    });
  });

  describe('CardHeader', () => {
    it('renders children correctly', () => {
      render(<CardHeader>Header content</CardHeader>);
      expect(screen.getByText('Header content')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      render(<CardHeader className="custom-header">Header content</CardHeader>);
      expect(screen.getByText('Header content')).toHaveClass('custom-header');
    });
  });

  describe('CardTitle', () => {
    it('renders children correctly', () => {
      render(<CardTitle>Title text</CardTitle>);
      expect(screen.getByRole('heading', { name: 'Title text' })).toBeInTheDocument();
    });

    it('applies custom className', () => {
      render(<CardTitle className="custom-title">Title text</CardTitle>);
      expect(screen.getByRole('heading', { name: 'Title text' })).toHaveClass('custom-title');
    });
  });

  describe('CardDescription', () => {
    it('renders children correctly', () => {
      render(<CardDescription>Description text</CardDescription>);
      // CardDescription renders a <p> tag.
      expect(screen.getByText('Description text')).toBeInTheDocument();
      expect(screen.getByText('Description text').tagName).toBe('P');
    });

    it('applies custom className', () => {
      render(<CardDescription className="custom-description">Description text</CardDescription>);
      expect(screen.getByText('Description text')).toHaveClass('custom-description');
    });
  });

  describe('CardContent', () => {
    it('renders children correctly', () => {
      render(<CardContent>Main content</CardContent>);
      expect(screen.getByText('Main content')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      render(<CardContent className="custom-content">Main content</CardContent>);
      expect(screen.getByText('Main content')).toHaveClass('custom-content');
    });
  });

  describe('CardFooter', () => {
    it('renders children correctly', () => {
      render(<CardFooter>Footer content</CardFooter>);
      expect(screen.getByText('Footer content')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      render(<CardFooter className="custom-footer">Footer content</CardFooter>);
      expect(screen.getByText('Footer content')).toHaveClass('custom-footer');
    });
  });
});

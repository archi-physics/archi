/**
 * Global type declarations for Playwright tests
 * 
 * Extends the Window interface with custom Archi utilities
 * that are attached at runtime by the application.
 */

interface ToastOptions {
  duration?: number;
}

interface ArchiUtils {
  escapeHtml(text: string | null | undefined): string;
  escapeAttr(text: string | null | undefined): string;
  formatSize(bytes: number): string;
  formatRelativeTime(dateString: string | null | undefined): string;
  isValidUrl(url: string): boolean;
  getFileIcon(filename: string): string;
  sanitizeUrl(url: string | null | undefined): string | null;
  debounce<T extends (...args: unknown[]) => unknown>(fn: T, delay: number): T;
}

interface Toast {
  success(message: string, options?: ToastOptions | number): string | void;
  error(message: string, options?: ToastOptions | number): string | void;
  dismiss(id?: string): void;
}

interface DataViewer {
  sanitizeUrl(url: string | null | undefined): string | null;
  escapeHtml(text: string | null | undefined): string;
}

declare global {
  interface Window {
    archiUtils?: ArchiUtils;
    toast?: Toast;
    dataViewer?: DataViewer;
  }
}

export {};


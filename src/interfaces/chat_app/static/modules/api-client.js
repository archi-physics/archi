/**
 * API Client Module
 * 
 * Centralized HTTP client for API requests.
 * Provides consistent error handling, retries, and response parsing.
 */

class ApiClient {
  constructor(options = {}) {
    this.options = {
      baseUrl: options.baseUrl || '',
      timeout: options.timeout || 30000,
      retries: options.retries || 0,
      retryDelay: options.retryDelay || 1000,
      headers: options.headers || {},
      ...options
    };
  }

  /**
   * Make an HTTP request
   * @param {string} endpoint - API endpoint
   * @param {Object} options - Fetch options
   * @returns {Promise<Object>} - Parsed response data
   */
  async request(endpoint, options = {}) {
    const url = this.options.baseUrl + endpoint;
    const method = options.method || 'GET';
    
    const fetchOptions = {
      method,
      headers: {
        ...this.options.headers,
        ...options.headers,
      },
      ...options,
    };

    // Add JSON body handling
    if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
      fetchOptions.headers['Content-Type'] = 'application/json';
      fetchOptions.body = JSON.stringify(options.body);
    }

    // Add timeout via AbortController
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || this.options.timeout);
    fetchOptions.signal = controller.signal;

    let lastError;
    const maxAttempts = (options.retries ?? this.options.retries) + 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const response = await fetch(url, fetchOptions);
        clearTimeout(timeoutId);

        // Parse response
        const data = await this.parseResponse(response);

        // Handle HTTP errors
        if (!response.ok) {
          const error = new ApiError(
            data.error || data.message || `HTTP ${response.status}`,
            response.status,
            data
          );
          
          // Don't retry client errors (4xx)
          if (response.status >= 400 && response.status < 500) {
            throw error;
          }
          
          // Retry server errors (5xx)
          lastError = error;
          if (attempt < maxAttempts) {
            await this.delay(this.options.retryDelay * attempt);
            continue;
          }
          throw error;
        }

        return data;
      } catch (error) {
        clearTimeout(timeoutId);
        
        // Handle abort/timeout
        if (error.name === 'AbortError') {
          throw new ApiError('Request timeout', 408);
        }
        
        // Handle network errors
        if (error instanceof TypeError && error.message.includes('fetch')) {
          lastError = new ApiError('Network error - server may be unavailable', 0);
          if (attempt < maxAttempts) {
            await this.delay(this.options.retryDelay * attempt);
            continue;
          }
          throw lastError;
        }
        
        // Re-throw API errors
        if (error instanceof ApiError) {
          throw error;
        }
        
        // Wrap unknown errors
        throw new ApiError(error.message || 'Unknown error', 0);
      }
    }

    throw lastError || new ApiError('Request failed after retries', 0);
  }

  /**
   * Parse response body
   * @param {Response} response - Fetch Response object
   * @returns {Promise<Object>} - Parsed data
   */
  async parseResponse(response) {
    const contentType = response.headers.get('content-type') || '';
    
    if (contentType.includes('application/json')) {
      try {
        return await response.json();
      } catch (e) {
        return { error: 'Invalid JSON response' };
      }
    }
    
    // Return text for non-JSON responses
    const text = await response.text();
    return { data: text };
  }

  /**
   * Delay helper for retries
   * @param {number} ms - Milliseconds to wait
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // HTTP method shortcuts
  async get(endpoint, options = {}) {
    return this.request(endpoint, { ...options, method: 'GET' });
  }

  async post(endpoint, body, options = {}) {
    return this.request(endpoint, { ...options, method: 'POST', body });
  }

  async put(endpoint, body, options = {}) {
    return this.request(endpoint, { ...options, method: 'PUT', body });
  }

  async delete(endpoint, options = {}) {
    return this.request(endpoint, { ...options, method: 'DELETE' });
  }

  /**
   * Upload a file
   * @param {string} endpoint - Upload endpoint
   * @param {File} file - File to upload
   * @param {Object} options - Additional options
   * @returns {Promise<Object>} - Upload response
   */
  async uploadFile(endpoint, file, options = {}) {
    const formData = new FormData();
    formData.append(options.fieldName || 'file', file);
    
    // Add additional fields
    if (options.fields) {
      for (const [key, value] of Object.entries(options.fields)) {
        formData.append(key, value);
      }
    }

    return this.request(endpoint, {
      method: 'POST',
      body: formData,
      timeout: options.timeout || 120000, // 2 min default for uploads
      ...options,
    });
  }
}

/**
 * Custom error class for API errors
 */
class ApiError extends Error {
  constructor(message, status, data = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }

  /**
   * Check if this is a "not found" error
   */
  get isNotFound() {
    return this.status === 404;
  }

  /**
   * Check if this is a server error
   */
  get isServerError() {
    return this.status >= 500;
  }

  /**
   * Check if this is a client error
   */
  get isClientError() {
    return this.status >= 400 && this.status < 500;
  }

  /**
   * Check if this is a network/connectivity error
   */
  get isNetworkError() {
    return this.status === 0;
  }

  /**
   * Get user-friendly error message
   */
  get userMessage() {
    if (this.isNetworkError) {
      return 'Unable to connect to the server. Please check your connection.';
    }
    if (this.isNotFound) {
      return 'The requested resource was not found.';
    }
    if (this.status === 408) {
      return 'Request timed out. Please try again.';
    }
    if (this.isServerError) {
      return 'Server error. Please try again later.';
    }
    return this.message;
  }
}

// Create default API client instance
const api = new ApiClient();

// Pre-configured clients for different services
const dataApi = new ApiClient({ retries: 1 });
const uploadApi = new ApiClient({ timeout: 120000 }); // 2 min timeout for uploads

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ApiClient, ApiError, api, dataApi, uploadApi };
}
if (typeof window !== 'undefined') {
  window.ApiClient = ApiClient;
  window.ApiError = ApiError;
  window.api = api;
  window.dataApi = dataApi;
  window.uploadApi = uploadApi;
}

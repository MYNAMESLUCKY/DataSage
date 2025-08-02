import streamlit as st
from typing import Dict, List, Any
import json

class DataSourceManager:
    """
    Manager for handling data source configurations and suggestions
    """
    
    def __init__(self):
        self.suggested_sources = self._load_suggested_sources()
    
    def _load_suggested_sources(self) -> List[Dict[str, Any]]:
        """Load predefined suggested data sources"""
        return [
            {
                'name': 'Wikipedia AI',
                'url': 'https://en.wikipedia.org/wiki/Artificial_intelligence',
                'type': 'web',
                'description': 'Comprehensive article about artificial intelligence',
                'category': 'Education'
            },
            {
                'name': 'ArXiv Recent Papers',
                'url': 'https://arxiv.org/list/cs.AI/recent',
                'type': 'web',
                'description': 'Recent AI research papers from ArXiv',
                'category': 'Research'
            },
            {
                'name': 'MIT Technology Review',
                'url': 'https://www.technologyreview.com/topic/artificial-intelligence/',
                'type': 'web',
                'description': 'Latest AI news and insights',
                'category': 'News'
            },
            {
                'name': 'OpenAI Blog',
                'url': 'https://openai.com/blog/',
                'type': 'web',
                'description': 'Updates and research from OpenAI',
                'category': 'Company Blog'
            },
            {
                'name': 'Machine Learning Mastery',
                'url': 'https://machinelearningmastery.com/',
                'type': 'web',
                'description': 'Practical machine learning tutorials',
                'category': 'Tutorial'
            },
            {
                'name': 'Towards Data Science',
                'url': 'https://towardsdatascience.com/',
                'type': 'web',
                'description': 'Data science and ML articles on Medium',
                'category': 'Community'
            }
        ]
    
    def get_suggested_sources(self) -> List[Dict[str, Any]]:
        """Get list of suggested data sources"""
        return self.suggested_sources
    
    def get_sources_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get suggested sources filtered by category"""
        return [
            source for source in self.suggested_sources 
            if source.get('category') == category
        ]
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        categories = set()
        for source in self.suggested_sources:
            categories.add(source.get('category', 'Other'))
        return sorted(list(categories))
    
    def add_custom_source(self, name: str, url: str, description: str = "", category: str = "Custom") -> Dict[str, Any]:
        """Add a custom data source"""
        custom_source = {
            'name': name,
            'url': url,
            'type': 'web',
            'description': description,
            'category': category,
            'custom': True
        }
        return custom_source
    
    def validate_source_url(self, url: str) -> Dict[str, Any]:
        """Validate a data source URL and return metadata"""
        import requests
        from urllib.parse import urlparse
        
        validation_result = {
            'valid': False,
            'accessible': False,
            'content_type': None,
            'title': None,
            'error': None
        }
        
        try:
            # Parse URL
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                validation_result['error'] = "Invalid URL format"
                return validation_result
            
            validation_result['valid'] = True
            
            # Test accessibility
            response = requests.head(url, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            validation_result['accessible'] = True
            validation_result['content_type'] = response.headers.get('content-type', '')
            
            # Try to get page title for HTML content
            if 'text/html' in validation_result['content_type']:
                try:
                    from bs4 import BeautifulSoup
                    html_response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(html_response.text, 'html.parser')
                    title_tag = soup.find('title')
                    if title_tag:
                        validation_result['title'] = title_tag.get_text().strip()
                except:
                    pass  # Title extraction failed, but that's OK
            
        except requests.RequestException as e:
            validation_result['error'] = f"Cannot access URL: {str(e)}"
        except Exception as e:
            validation_result['error'] = f"Validation error: {str(e)}"
        
        return validation_result
    
    def render_source_selector(self) -> List[Dict[str, Any]]:
        """Render interactive source selector UI"""
        st.subheader("Data Source Selection")
        
        selected_sources = []
        
        # Tabs for different source types
        tab1, tab2, tab3 = st.tabs(["Suggested Sources", "Custom URL", "Bulk Import"])
        
        with tab1:
            self._render_suggested_sources_tab(selected_sources)
        
        with tab2:
            self._render_custom_url_tab(selected_sources)
        
        with tab3:
            self._render_bulk_import_tab(selected_sources)
        
        return selected_sources
    
    def _render_suggested_sources_tab(self, selected_sources: List[Dict[str, Any]]):
        """Render suggested sources tab"""
        st.write("Choose from our curated list of high-quality data sources:")
        
        # Category filter
        categories = ['All'] + self.get_categories()
        selected_category = st.selectbox("Filter by Category", categories, key="category_filter")
        
        # Filter sources
        if selected_category == 'All':
            sources_to_show = self.suggested_sources
        else:
            sources_to_show = self.get_sources_by_category(selected_category)
        
        # Display sources
        for i, source in enumerate(sources_to_show):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{source['name']}**")
                    st.caption(source.get('description', ''))
                    st.caption(f"URL: {source['url']}")
                
                with col2:
                    st.caption(f"Category: {source.get('category', 'Other')}")
                
                with col3:
                    if st.button("Add", key=f"add_suggested_{i}"):
                        selected_sources.append(source)
                        st.success(f"Added {source['name']}")
                        st.rerun()
    
    def _render_custom_url_tab(self, selected_sources: List[Dict[str, Any]]):
        """Render custom URL tab"""
        st.write("Add your own data source by providing a URL:")
        
        with st.form("custom_source_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                custom_url = st.text_input(
                    "URL",
                    placeholder="https://example.com/article",
                    help="Enter the URL of the webpage you want to add as a data source"
                )
            
            with col2:
                custom_name = st.text_input(
                    "Name (Optional)",
                    placeholder="My Custom Source",
                    help="Give your source a descriptive name"
                )
            
            custom_description = st.text_area(
                "Description (Optional)",
                placeholder="Brief description of this data source...",
                height=100
            )
            
            submit_custom = st.form_submit_button("Add Custom Source")
            
            if submit_custom and custom_url:
                # Validate URL
                validation = self.validate_source_url(custom_url)
                
                if validation['valid']:
                    source_name = custom_name or validation.get('title') or f"Source from {custom_url}"
                    
                    custom_source = self.add_custom_source(
                        name=source_name,
                        url=custom_url,
                        description=custom_description
                    )
                    
                    selected_sources.append(custom_source)
                    
                    if validation['accessible']:
                        st.success(f"✅ Added custom source: {source_name}")
                    else:
                        st.warning(f"⚠️ Added source but couldn't verify accessibility: {source_name}")
                    
                    st.rerun()
                else:
                    st.error(f"❌ Invalid URL: {validation.get('error', 'Unknown error')}")
    
    def _render_bulk_import_tab(self, selected_sources: List[Dict[str, Any]]):
        """Render bulk import tab"""
        st.write("Import multiple URLs at once:")
        
        urls_text = st.text_area(
            "URLs (one per line)",
            placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com",
            height=150,
            help="Enter multiple URLs, one per line. Empty lines will be ignored."
        )
        
        if st.button("Import URLs"):
            if urls_text:
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                
                if urls:
                    st.write(f"Processing {len(urls)} URLs...")
                    
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, url in enumerate(urls):
                        validation = self.validate_source_url(url)
                        
                        if validation['valid']:
                            source_name = validation.get('title') or f"Source from {url}"
                            
                            custom_source = self.add_custom_source(
                                name=source_name,
                                url=url,
                                description=f"Bulk imported from {url}"
                            )
                            
                            selected_sources.append(custom_source)
                            results.append(f"✅ {source_name}")
                        else:
                            results.append(f"❌ {url}: {validation.get('error', 'Invalid')}")
                        
                        progress_bar.progress((i + 1) / len(urls))
                    
                    # Show results
                    st.subheader("Import Results")
                    for result in results:
                        st.write(result)
                    
                    st.rerun()
                else:
                    st.warning("No valid URLs found in the input.")
    
    def get_source_statistics(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about selected sources"""
        if not sources:
            return {}
        
        stats = {
            'total_sources': len(sources),
            'categories': {},
            'domains': {},
            'source_types': {}
        }
        
        for source in sources:
            # Count by category
            category = source.get('category', 'Other')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Count by domain
            try:
                from urllib.parse import urlparse
                domain = urlparse(source['url']).netloc
                stats['domains'][domain] = stats['domains'].get(domain, 0) + 1
            except:
                pass
            
            # Count by type
            source_type = source.get('type', 'web')
            stats['source_types'][source_type] = stats['source_types'].get(source_type, 0) + 1
        
        return stats
    
    def export_sources_config(self, sources: List[Dict[str, Any]]) -> str:
        """Export sources configuration as JSON"""
        config = {
            'sources': sources,
            'exported_at': st.session_state.get('current_time', 'unknown'),
            'version': '1.0'
        }
        return json.dumps(config, indent=2)
    
    def import_sources_config(self, config_json: str) -> List[Dict[str, Any]]:
        """Import sources configuration from JSON"""
        try:
            config = json.loads(config_json)
            return config.get('sources', [])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {str(e)}")

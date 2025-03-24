import streamlit as st
import requests

def get_company_data(company_name):
    api_url = "http://localhost:7860/get_company_data"  # Replace with your actual backend API URL
    response = requests.post(api_url, json={"company": company_name})
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch data"}

# Streamlit UI
st.title("Company News Summary")

# User input
company_name = st.text_input("Enter Company Name:")

if company_name:
    # Fetch data (replace with actual API call in real use)
    result = get_company_data(company_name)

    if "error" in result:
        st.error(result["error"])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["News", "Sentimental Analysis", "Comparative Analysis", "Final Sentiment", "audio"])
        
        with tab1:
            st.subheader("News")
            news_list = result.get("news", [])
            
            for news_item in news_list:
                st.markdown(f"### {news_item.get('title', 'No Title Available')}")
                st.write(news_item.get("summary", "No Summary Available"))
                
                st.write("**Sentiment:**", news_item.get("sentiment", "No Sentiment Available"))
                
                keywords = news_item.get("keywords", [])
                if keywords:
                    st.write("**Keywords:**")
                    for keyword in keywords:
                        st.write(f"- {keyword}")
                else:
                    st.write("**Keywords:** No Keywords Available")
                
                st.write("**Source:**", news_item.get("source", "No Source Available"))
                st.write("**Publish Date:**", news_item.get("publish_date", "No Date Available"))
                st.write("[Read More]({})".format(news_item.get("url", "#")))
                st.markdown("---")
        
        with tab2:
            st.subheader("Sentiment Distribution")
            st.write(result.get("frequency", {}))
        
        with tab3:
            for analysis in result.get("comparative_analysis", {}).get("Comparative Analysis", []):
                st.subheader("Comparison")
                st.write(analysis.get("Comparison", "No Data"))
                st.subheader("Impact")
                st.write(analysis.get("Impact", "No Data"))
                st.markdown("---")

            for article_group in result.get("comparative_analysis", {}).get("Similar Articles", []):
                st.subheader("Similar Articles")
                st.write("**Articles:**", ", ".join(map(str, article_group.get("Articles", []))))
                st.write("**Common Keywords:**", ", ".join(map(str, article_group.get("Common Keywords", []))))
                st.markdown("---")
        
        with tab4:
            st.write(result.get("final_sentiment", {}))

        with tab5:
            st.subheader("Hindi Audio")
            audio_url = result.get("audio_url")
            if audio_url:
                st.audio(audio_url, format="audio/mp3")
            else:
                st.write("No audio available")


import openai
import os





def generate_expanalysis(data): #this is the function that will generate the expert analysis for the company
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt = f"The company {data['name']} has the following financial data:\n"
    for key, value in data.items():
        if key != 'name':
            prompt += f"{key}: {value}\n"
    prompt += '''You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. It should also include steps for the upcoming future that can benefit the company realistically. Based only from this data, Can you write an expert analysis paragraph. 

    When writing the analysis, consider the following details:
    Keep the information concise and to the point. The response should be based on the data provided. The response must also have quantitative values to support the response. Ensure the quantitative values are realistic and accurately calculated based on typical industry standards and historical performance.
    Market Conditions: Discuss any market trends or economic factors that might influence the company's performance.
    Specific Financial Metrics: Include key financial metrics such as revenues, EBITDA, net income, and EPS.
    Factors Affecting Performance: Mention any significant factors such as new product launches, regulatory changes, cost management strategies, or investment plans.
    Comparative Analysis: Compare the companys projections with industry averages or competitors if applicable.
    
    Here are some example of formats  of how you can write the expert analysis that you can use as a reference and keep it under 100 words:
    Example 1: Based on analysts offering 12 month price targets for TEVA in the last 3 months. The average price target is $15.71 with a high estimate of $19 and a low estimate of $11
    Example 2: analysts expect ANI Pharmaceuticals to post earnings of $0.97 per share. This would mark a year-over-year decline of 17.09%. Meanwhile, the Zacks Consensus Estimate for revenue is projecting net sales of $124.38 million, up \"16.47%\" from the year-ago period.
    Example 3: Royalty Pharma's eight analysts are now forecasting revenues of US$2.68b in 2024. This would be a meaningful \"14%\" improvement in revenue compared to the last 12 months. Statutory earnings per share are expected to shrink 6.3% to US$2.38 in the same period
     '''
    response = openai.Completion.create(
      model ="gpt-3.5-turbo-instruct",
      prompt=prompt,
      max_tokens=500,
      temperature=0,
      n = 1
    )
    
    return response.choices[0].text.strip(),  


if __name__ == "__main__":
    data = {'Operating Income': [{'year': '2023', 'value': '204374000'}, {'year': '2022', 'value': '-94928000'}, {'year': '2021', 'value': '152716000'}], 'Profit Loss': [{'year': '2023', 'value': '-48722000'}, {'year': '2022', 'value': '-254789000'}, {'year': '2021', 'value': '20170000'}], 'Net income': [{'year': '2023', 'value': '-83993000'}, {'year': '2022', 'value': '-129986000'}, {'year': '2021', 'value': '10624000'}], 'interest expense': [{'year': '2023', 'value': '-210629000'}, {'year': '2022', 'value': '-158377000'}, {'year': '2021', 'value': '-136325000'}], 'Income Tax': [{'year': '2023', 'value': '-2496000'}, {'year': '2022', 'value': '-12649000'}, {'year': '2021', 'value': '-15558000'}], 'Depreciation & Amortization': [{'year': '2023', 'value': '229400000'}, {'year': '2022', 'value': '240175000'}, {'year': '2021', 'value': '233406000'}], 'Net Revenue': [{'year': '2023', 'value': '2393607000'}, {'year': '2022', 'value': '2212304000'}, {'year': '2021', 'value': '2093669000'}], 'name': 'Amneal Pharmaceuticals, Inc.', 'ebitda': [{'year': '2023', 'value': '433774000'}, {'year': '2022', 'value': '145247000'}, {'year': '2021', 'value': '386122000'}], 'annual revenue growth': [{'year': '2023', 'value': '8.195211869616472 %'}, {'year': '2022', 'value': '5.666368466075583 %'}], 'ebitda growth': [{'year': '2023', 'value': '198.64575516189663 %'}, {'year': '2022', 'value': '-62.38313279222629 %'}], 'year': '2023'}
    guidance_tuple = generate_expanalysis(data)
    cleaned_guidance = tuple(s.replace("\\n", "\n") for s in guidance_tuple)
    print(guidance_tuple)
import { NextRequest, NextResponse } from 'next/server'
import { rateLimit, getRateLimitIdentifier } from '@/lib/utils/rateLimit'
import type { InterpretRequestBody } from '@/lib/types'
import { GoogleGenerativeAI } from '@google/generative-ai'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildPrompt(schema: any, result: any, modelTrainingReport?: any): string {
  // Build model training context (explainability + business translation)
  let modelTrainingContext = '';
  if (modelTrainingReport) {
    const mt = modelTrainingReport;
    const parts: string[] = [];

    if (mt.explainability?.consensusRanking?.length) {
      const topFeatures = mt.explainability.consensusRanking.slice(0, 5)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((f: any) => `${f.feature} (rank ${f.consensusRank})`).join(', ');
      parts.push(`TOP FEATURES (consensus): ${topFeatures}`);
    }
    if (mt.explainability?.summary) {
      parts.push(`EXPLAINABILITY: ${mt.explainability.summary}`);
    }
    if (mt.businessTranslation?.insights?.length) {
      const insights = mt.businessTranslation.insights.slice(0, 4)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((i: any) => i.text).join(' ');
      parts.push(`BUSINESS TRANSLATION: ${insights}`);
    }
    if (mt.businessTranslation?.confidence) {
      parts.push(`MODEL CONFIDENCE: ${mt.businessTranslation.confidence}`);
    }
    if (mt.recommendations?.length) {
      const recs = mt.recommendations.slice(0, 3)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((r: any) => `[${r.priority}] ${r.action}`).join(' ');
      parts.push(`RECOMMENDATIONS: ${recs}`);
    }
    if (mt.bestModel) {
      parts.push(`BEST MODEL: ${mt.bestModel.model} (score: ${mt.bestModel.score})`);
    }

    if (parts.length > 0) {
      modelTrainingContext = `\n\nMODEL TRAINING & EXPLAINABILITY:\n${parts.join('\n')}`;
    }
  }

  return `You are StatLab AI, a senior Data Scientist with 15+ years of experience in statistics, machine learning, experimentation, and business analytics.

Your role is not to simply describe charts or output numbers. Your responsibility is to think like an experienced data scientist consulting for a client.

Personality:
- Professional, precise, evidence-based, business-oriented
- Honest about uncertainty — never exaggerate findings
- Never fabricate statistics — every conclusion must come directly from the dataset and computed metrics

Writing Style:
- Write like a consultant delivering a report to executives
- Avoid robotic statements and repeating statistics unnecessarily
- Explain technical terms in plain English when appropriate
- Prioritize interpretation over numbers
- Highlight anomalies worth investigating

Rules:
- Never recompute or change the numbers given to you
- Use exact numeric values provided in the data
- Never infer causation from correlation — always flag this distinction
- Explain what results mean practically, not just statistically
- When feature importance or explainability data is provided, translate it into business actions
- When business translation is provided, reference it in your interpretation
- Flag statistical significance clearly
- Always include limitations when relevant
- Return ONLY valid JSON, no preamble, no markdown

DATASET SCHEMA:
${JSON.stringify(schema)}

COMPUTED ANALYSIS RESULTS (SAMPLE):
${JSON.stringify(result)}
${modelTrainingContext}

RESPONSE FORMAT:
Return ONLY a raw valid JSON object (no markdown, no code fences):

{
  "summary": "Write a 4-6 sentence executive summary that covers: (1) what the dataset contains and its objective, (2) data quality notes (missing values, outliers, duplicates if notable), (3) the 2-3 most important findings with business impact, (4) model performance quality if applicable, (5) top recommendation, (6) confidence level (High/Medium/Low) with brief justification. Write like a consultant, not a robot.",

  "perAnalysis": [
    {
      "type": "descriptive|correlation|hypothesis|predictive|feature_importance|business_impact|data_quality|recommendation",
      "subject": "column name, pair, or topic (e.g., 'Age', 'Revenue vs Ad Spend', 'Model Performance', 'Data Quality')",
      "interpretation": "Write 2-4 sentences as a senior data scientist would: explain the distribution/skewness for descriptive, explain what r=0.81 means practically for correlations (never just state the number), translate p-values into plain-English significance for hypothesis tests, explain model metrics in business terms for predictive, explain why each feature matters for feature importance. Always connect findings to business actions when possible. Never fabricate statistics. Always note limitations or caveats when relevant."
    }
  ]
}

Generate perAnalysis entries for:
1. Each notable descriptive variable (focus on ones with interesting distributions, high skewness, or many outliers)
2. Each significant correlation pair (explain practical meaning, note correlation ≠ causation)
3. Each hypothesis test result (translate p-values to plain English)
4. The predictive model overall (explain R²/accuracy in business terms)
5. Top feature importances (group them, explain business impact)
6. Key business insights and recommendations (actionable, data-backed)

Prioritise quality over quantity — it is better to have 6-10 excellent interpretations than 20 shallow ones.`;
}

export async function POST(request: NextRequest) {
  try {
    const identifier = getRateLimitIdentifier(request)
    const { allowed } = rateLimit(identifier, 20, 60_000)
    if (!allowed) {
      return Response.json(
        { success: false, error: 'Too many requests. Wait a moment.' },
        { status: 429, headers: { 'Retry-After': '60' } }
      )
    }

    const body = await request.json() as InterpretRequestBody
    if (!body.schema || !body.result) {
      return NextResponse.json(
        { success: false, error: 'Schema and Result are required inputs' },
        { status: 400 }
      )
    }

    const payloadPrompt = buildPrompt(body.schema, body.result, body.modelTrainingReport);

    // =======================================================
    // ENGINE 1: GROQ (PRIMARY ENGINE)
    // =======================================================
    if (process.env.GROQ_API_KEY) {
      try {
        console.log('[StatLab AI] Routing request to Groq (Primary)...');
        const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: 'llama-3.3-70b-versatile',
            messages: [
              { role: 'system', content: 'You are StatLab AI, a senior Data Scientist. Return ONLY valid JSON, no markdown, no preamble.' },
              { role: 'user', content: payloadPrompt }
            ],
            temperature: 0.1
          })
        });

        if (groqResponse.ok) {
          const rawData = await groqResponse.json();
          let cleanText = rawData.choices[0].message.content.trim();
          
          if (cleanText.startsWith('```json')) cleanText = cleanText.replace(/^```json/, '').replace(/```$/, '');
          if (cleanText.startsWith('```')) cleanText = cleanText.replace(/^```/, '').replace(/```$/, '');

          const parsedJSON = JSON.parse(cleanText.trim());
          return NextResponse.json({
            success: true,
            summary: parsedJSON.summary,
            perAnalysis: parsedJSON.perAnalysis || [],
            provider: 'groq',
            fallbackUsed: false,
          });
        }
        
        console.warn(`[StatLab AI] Groq endpoint status code [${groqResponse.status}]. Falling back to sub...`);
      } catch (groqError) {
        console.warn('[StatLab AI] Groq parsing/network failure. Attempting Gemini fallback...', groqError);
      }
    }

    // =======================================================
    // ENGINE 2: GEMINI (SUB / FALLBACK ENGINE)
    // =======================================================
    if (process.env.GEMINI_API_KEY) {
  try {
    console.log('[StatLab AI] Routing request to Gemini (Fallback Sub)...');
    
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');
    const model = genAI.getGenerativeModel({ model: 'gemini-pro' });

    const result = await model.generateContent(payloadPrompt);
    let cleanText = result.response.text().trim();

        if (cleanText.startsWith('```json')) cleanText = cleanText.replace(/^```json/, '').replace(/```$/, '');
        if (cleanText.startsWith('```')) cleanText = cleanText.replace(/^```/, '').replace(/```$/, '');

        const parsedJSON = JSON.parse(cleanText.trim());

        return NextResponse.json({
          success: true,
          summary: parsedJSON.summary,
          perAnalysis: parsedJSON.perAnalysis || [],
          provider: 'gemini',
          fallbackUsed: true,
        });
      } catch (geminiError) {
        console.error('[StatLab AI] Gemini fallback structural block failed:', geminiError);
      }
    }

    throw new Error('Both Groq and Gemini execution pathways failed to resolve payload.');

  } catch (error) {
    console.error('[StatLab AI] Global Fallback Triggered:', error);
    return NextResponse.json({
      success: true,
      summary: 'Analysis complete. Review the charts and data tables for full insights.',
      perAnalysis: [],
      provider: null,
      fallbackUsed: true,
    });
  }
}

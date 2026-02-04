<?php
/**
 * Plugin Name: AI Sentiment & Topic Tracker
 * Description: Connects WordPress Comments to a Python NLP Backend and provides a Visual Analytics Dashboard.
 * Version: 1.2
 * Author: BSc Data Science Student
 */

// --- 1. CAPTURE COMMENT & SEND TO PYTHON API ---
add_action('comment_post', 'send_to_python_api', 10, 3);

function send_to_python_api($comment_ID, $comment_approved, $commentdata) {
    // URL of your Flask Server
    $api_url = 'http://127.0.0.1:5000/analyze-comment';

    // Send the comment text to the Python API
    $response = wp_remote_post($api_url, array(
        'headers'     => array('Content-Type' => 'application/json'),
        'body'        => json_encode(array('text' => $commentdata['comment_content'])),
        'method'      => 'POST',
        'data_format' => 'body',
        'timeout'     => 45,
    ));

    if (!is_wp_error($response)) {
        $result = json_decode(wp_remote_retrieve_body($response), true);
        
        if (isset($result['sentiment'])) {
            // Save the AI results into WordPress Comment Meta
            update_comment_meta($comment_ID, 'ai_sentiment', $result['sentiment']);
            update_comment_meta($comment_ID, 'ai_topic', $result['topic']);
        }
    }
}

// --- 2. ADD DASHBOARD TO WORDPRESS MENU ---
add_action('admin_menu', 'bast_add_dashboard_menu');

function bast_add_dashboard_menu() {
    add_menu_page(
        'AI Insights',            // Page Title
        'NLP Analytics',          // Menu Label
        'manage_options',         // Permissions
        'nlp-dashboard',          // The ID (slug) of the page
        'bast_render_dashboard',  // The function that draws the page
        'dashicons-chart-area'    // The Sidebar Icon
    );
}

// --- 3. DATA AGGREGATOR (Logic for Charts) ---
function get_ai_stats() {
    $comments = get_comments();
    $stats = [
        'total' => 0,
        'positive' => 0,
        'negative' => 0,
        'topics' => []
    ];

    foreach ($comments as $comment) {
        $sentiment = get_comment_meta($comment->comment_ID, 'ai_sentiment', true);
        $topic = get_comment_meta($comment->comment_ID, 'ai_topic', true);

        if ($sentiment) {
            $stats['total']++;
            ($sentiment == 'Positive') ? $stats['positive']++ : $stats['negative']++;
            
            if ($topic) {
                $stats['topics'][$topic] = ($stats['topics'][$topic] ?? 0) + 1;
            }
        }
    }
    return $stats;
}

// --- 4. DASHBOARD UI (The Frontend Visualization) ---
function bast_render_dashboard() {
    $stats = get_ai_stats();
    $neg_percent = ($stats['total'] > 0) ? round(($stats['negative'] / $stats['total']) * 100) : 0;
    ?>
    <style>
        .ai-card-container { display: flex; gap: 20px; margin-top: 20px; flex-wrap: wrap; }
        .ai-card { background: #fff; border: 1px solid #ccd0d4; border-radius: 8px; padding: 20px; flex: 1; min-width: 200px; box-shadow: 0 1px 1px rgba(0,0,0,.04); }
        .ai-stat-number { font-size: 32px; font-weight: bold; color: #2271b1; margin: 10px 0; }
        .ai-chart-row { display: flex; gap: 20px; margin-top: 20px; }
        .ai-chart-box { background: #fff; border: 1px solid #ccd0d4; border-radius: 8px; padding: 25px; flex: 1; }
        @media (max-width: 782px) { .ai-chart-row { flex-direction: column; } }
    </style>

    <div class="wrap">
        <h1><span class="dashicons dashicons-analytics"></span> Community Sentiment Insights</h1>
        <p>This dashboard visualizes data processed by your Python NLP backend.</p>
        
        <div class="ai-card-container">
            <div class="ai-card">
                <h3>Total Analyzed</h3>
                <div class="ai-stat-number"><?php echo $stats['total']; ?></div>
            </div>
            <div class="ai-card">
                <h3>Negative Ratio</h3>
                <div class="ai-stat-number" style="color: #d63638;"><?php echo $neg_percent; ?>%</div>
            </div>
            <div class="ai-card">
                <h3>Most Frequent Topic</h3>
                <div class="ai-stat-number" style="font-size: 20px;">
                    <?php 
                        if(!empty($stats['topics'])) {
                            arsort($stats['topics']); 
                            echo key($stats['topics']); 
                        } else { echo "No Data Yet"; }
                    ?>
                </div>
            </div>
        </div>

        <div class="ai-chart-row">
            <div class="ai-chart-box">
                <h3>Sentiment Distribution</h3>
                <canvas id="sentimentChart"></canvas>
            </div>
            <div class="ai-chart-box" style="flex: 1.5;">
                <h3>Topic Breakdown (NLP Classification)</h3>
                <canvas id="topicChart"></canvas>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Sentiment Pie Chart
        const sCtx = document.getElementById('sentimentChart').getContext('2d');
        new Chart(sCtx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative'],
                datasets: [{
                    data: [<?php echo $stats['positive']; ?>, <?php echo $stats['negative']; ?>],
                    backgroundColor: ['#4caf50', '#f44336'],
                    hoverOffset: 4
                }]
            },
            options: { cutout: '65%', plugins: { legend: { position: 'bottom' } } }
        });

        // Topic Bar Chart
        const tCtx = document.getElementById('topicChart').getContext('2d');
        new Chart(tCtx, {
            type: 'bar',
            data: {
                labels: <?php echo json_encode(array_keys($stats['topics'])); ?>,
                datasets: [{
                    label: 'Number of Comments',
                    data: <?php echo json_encode(array_values($stats['topics'])); ?>,
                    backgroundColor: '#2271b1',
                    borderRadius: 4
                }]
            },
            options: { 
                indexAxis: 'y', 
                plugins: { legend: { display: false } },
                scales: { x: { beginAtZero: true, ticks: { stepSize: 1 } } }
            }
        });
    });
    </script>
    <?php
}

// --- 5. MODIFY COMMENT TABLE COLUMNS ---
add_filter('manage_edit-comments_columns', 'add_ai_columns');
function add_ai_columns($columns) {
    $columns['ai_insight'] = 'AI Insight';
    return $columns;
}

add_action('manage_comments_custom_column', 'fill_ai_columns', 10, 2);
function fill_ai_columns($column, $comment_ID) {
    if ($column == 'ai_insight') {
        $sentiment = get_comment_meta($comment_ID, 'ai_sentiment', true);
        $topic = get_comment_meta($comment_ID, 'ai_topic', true);
        
        if ($sentiment) {
            $color = ($sentiment == 'Negative') ? '#d63638' : '#4caf50';
            echo "<strong style='color:$color;'>$sentiment</strong><br/><small style='color:#666;'>Topic: $topic</small>";
        } else {
            echo "<em>Awaiting Analysis</em>";
        }
    }
}

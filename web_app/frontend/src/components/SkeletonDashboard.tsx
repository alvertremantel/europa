export function SkeletonDashboard() {
  return (
    <section className="dashboard-skeleton" aria-hidden="true">
      {Array.from({ length: 5 }, (_, index) => (
        <div key={index} className="dashboard-skeleton__card" />
      ))}
    </section>
  )
}
